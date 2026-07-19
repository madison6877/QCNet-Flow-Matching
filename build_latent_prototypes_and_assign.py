# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, contextlib, csv, heapq, json, math, random, shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm
from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM
try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as exc:
    raise ImportError("缺少 scikit-learn，请先执行：pip install scikit-learn") from exc

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_device(name:str)->torch.device:
    return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else ("cpu" if name=="auto" else name))

def autocast_context(device:torch.device,precision:str):
    if precision=="32" or device.type!="cuda": return contextlib.nullcontext()
    return torch.autocast(device_type="cuda",dtype=torch.bfloat16 if precision=="bf16" else torch.float16)

def load_checkpoint(path:str)->Dict[str,Any]:
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")

@torch.no_grad()
def encode_centered_raw(model:QCNetFM,target:torch.Tensor,predict_mask:torch.Tensor)->torch.Tensor:
    encoded=model.latent_encoder.vae.encode(target,predict_mask=predict_mask); mu=encoded[0] if isinstance(encoded,(tuple,list)) else encoded
    if mu.ndim==2: z_raw=mu.unsqueeze(1)
    elif mu.ndim==3 and mu.size(0)==int(model.vae_num_intents) and mu.size(1)==target.size(0): z_raw=mu.transpose(0,1).contiguous()
    elif mu.ndim==3 and mu.size(0)==target.size(0): z_raw=mu
    else: raise RuntimeError(f"VAE posterior mean 形状无法识别：mu={tuple(mu.shape)}, target={tuple(target.shape)}")
    z=z_raw.float()-model.z_mean.to(device=z_raw.device,dtype=torch.float32)
    if z.ndim!=3 or z.size(1)!=1 or z.size(-1)!=model.latent_dim: raise RuntimeError(f"latent 形状错误：{tuple(z.shape)}")
    return z

def get_cached_targets(model:QCNetFM,batch:Mapping[str,Any],device:torch.device,agent_scope:str):
    if "target" not in batch or "predict_mask" not in batch: raise KeyError(f"VAE cache batch 必须包含 target 和 predict_mask，实际 keys={list(batch.keys())}")
    target=batch["target"].to(device=device,dtype=torch.float32,non_blocking=True)[...,:model.output_dim]
    predict_mask=batch["predict_mask"].to(device=device,non_blocking=True).bool()
    if target.ndim!=3 or predict_mask.ndim!=2 or target.shape[:2]!=predict_mask.shape: raise ValueError(f"VAE cache target/mask 形状错误：target={tuple(target.shape)}, mask={tuple(predict_mask.shape)}")
    valid=predict_mask.any(dim=-1)
    if agent_scope=="focal":
        category=batch.get("category",None)
        if category is None: raise RuntimeError("VAE 缓存没有 category，不能用于 --agent_scope focal。")
        valid=valid&category.to(device=device,non_blocking=True).long().view(-1).eq(3)
    return target,predict_mask,valid

def _fill_missing_positions(position:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    n,t,_=position.shape; ids=torch.arange(t,device=position.device).view(1,t).expand(n,t)
    previous=torch.where(mask,ids,torch.full_like(ids,-1)).cummax(dim=1).values
    first=torch.where(mask,ids,torch.full_like(ids,t)).min(dim=1).values
    gather_idx=torch.where(previous.ge(0),previous,first[:,None]).clamp(0,t-1)
    return position.gather(1,gather_idx[...,None].expand(n,t,position.size(-1)))

def _segment_edges(num_steps:int,num_segments:int)->List[Tuple[int,int]]:
    edges=torch.linspace(0,num_steps,num_segments+1).round().long().tolist(); result=[]
    for i in range(num_segments):
        s,e=int(edges[i]),int(edges[i+1]); e=max(e,s+1); result.append((s,min(e,num_steps)))
    return result

def _dct_matrix(num_steps:int,num_coeffs:int,device:torch.device,dtype:torch.dtype)->torch.Tensor:
    n=torch.arange(num_steps,device=device,dtype=dtype).unsqueeze(1); k=torch.arange(num_coeffs,device=device,dtype=dtype).unsqueeze(0)
    matrix=torch.cos(math.pi/num_steps*(n+0.5)*k); scale=torch.full((num_coeffs,),math.sqrt(2.0/num_steps),device=device,dtype=dtype); scale[0]=math.sqrt(1.0/num_steps)
    return matrix*scale.unsqueeze(0)

@torch.no_grad()
def build_trajectory_descriptor_blocks(target_normalized:torch.Tensor,predict_mask:torch.Tensor,trajectory_scale:float,dct_coeffs:int,num_segments:int,time_step:float,slow_speed_threshold:float)->Dict[str,torch.Tensor]:
    if target_normalized.ndim!=3 or predict_mask.ndim!=2: raise ValueError("descriptor 输入形状错误。")
    position=target_normalized.float()*float(trajectory_scale); mask=predict_mask.bool(); position=_fill_missing_positions(position,mask)
    n,t,_=position.shape; origin=torch.zeros((n,1,2),device=position.device,dtype=position.dtype)
    delta=position-torch.cat([origin,position[:,:-1]],dim=1); speed=delta.norm(dim=-1)/float(time_step)
    step_valid=mask; speed_valid=torch.where(step_valid,speed,torch.zeros_like(speed))
    dct=_dct_matrix(t,dct_coeffs,position.device,position.dtype); coeff=torch.einsum("ntd,tk->nkd",position,dct)
    shape_dct=torch.cat([coeff[:,:,0],coeff[:,:,1]],dim=-1)
    segment_displacement=[]; segment_speed=[]; segment_heading=[]
    eps=1e-8; prev_delta=torch.cat([delta[:,:1],delta[:,:-1]],dim=1)
    cross=prev_delta[...,0]*delta[...,1]-prev_delta[...,1]*delta[...,0]; dot=(prev_delta*delta).sum(dim=-1)
    turn=torch.atan2(cross,dot.clamp(min=-1e12,max=1e12)); moving=(prev_delta.norm(dim=-1)/time_step>slow_speed_threshold)&(speed>slow_speed_threshold)&mask
    turn=torch.where(moving,turn,torch.zeros_like(turn)); edges=_segment_edges(t,num_segments); previous_endpoint=origin[:,0]
    for s,e in edges:
        endpoint=position[:,e-1]; segment_displacement.append(endpoint-previous_endpoint); previous_endpoint=endpoint
        seg_mask=mask[:,s:e]; denom=seg_mask.sum(dim=1).clamp_min(1).float(); segment_speed.append((speed_valid[:,s:e]*seg_mask.float()).sum(dim=1)/denom)
        segment_heading.append(turn[:,s:e].sum(dim=1))
    segment_displacement=torch.cat(segment_displacement,dim=-1); segment_speed=torch.stack(segment_speed,dim=-1); segment_heading=torch.stack(segment_heading,dim=-1)
    time_ids=torch.arange(t,device=position.device).view(1,t); last_valid=time_ids.masked_fill(~mask,-1).max(dim=1).values.clamp_min(0)
    endpoint=position.gather(1,last_valid[:,None,None].expand(n,1,2)).squeeze(1); final_speed=speed.gather(1,last_valid[:,None]).squeeze(1)
    path_length=(speed_valid*float(time_step)).sum(dim=1); total_heading=turn.sum(dim=1)
    abs_y=position[...,1].abs().masked_fill(~mask,-1); max_y_idx=abs_y.argmax(dim=1); signed_lateral=position[...,1].gather(1,max_y_idx[:,None]).squeeze(1)
    valid_count=mask.sum(dim=1).clamp_min(1).float(); slow_ratio=((speed<slow_speed_threshold)&mask).sum(dim=1).float()/valid_count
    global_features=torch.stack([endpoint[:,0],endpoint[:,1],final_speed,path_length,total_heading,signed_lateral,slow_ratio],dim=-1)
    return {"shape_dct":shape_dct,"segment_displacement":segment_displacement,"segment_speed":segment_speed,"segment_heading":segment_heading,"global":global_features}

def descriptor_block_weights(args)->Dict[str,float]:
    return {"shape_dct":args.descriptor_weight_dct,"segment_displacement":args.descriptor_weight_displacement,"segment_speed":args.descriptor_weight_speed,"segment_heading":args.descriptor_weight_heading,"global":args.descriptor_weight_global}

def fit_descriptor_normalizer(blocks:Dict[str,torch.Tensor],weights:Dict[str,float],eps:float=1e-6)->Dict[str,Any]:
    stats={"block_order":list(blocks.keys()),"weights":dict(weights),"mean":{},"std":{},"dim":{}}
    for name,x in blocks.items():
        xf=x.float(); stats["mean"][name]=xf.mean(dim=0); stats["std"][name]=xf.std(dim=0,unbiased=False).clamp_min(eps); stats["dim"][name]=int(x.size(1))
    return stats

def transform_descriptor_blocks(blocks:Dict[str,torch.Tensor],stats:Dict[str,Any])->torch.Tensor:
    outputs=[]
    for name in stats["block_order"]:
        x=blocks[name].float(); mean=stats["mean"][name].to(x); std=stats["std"][name].to(x); dim=int(stats["dim"][name]); weight=float(stats["weights"][name])
        outputs.append((x-mean)/std*math.sqrt(weight/max(dim,1)))
    return torch.cat(outputs,dim=-1)

def _sample_numpy(x:torch.Tensor,max_samples:int,seed:int)->np.ndarray:
    if max_samples>0 and x.size(0)>max_samples:
        rng=np.random.default_rng(seed); idx=torch.from_numpy(rng.choice(x.size(0),size=max_samples,replace=False)).long(); return x[idx].numpy()
    return x.numpy()

def fit_coarse_kmeans(descriptor:torch.Tensor,args)->Dict[str,Any]:
    if descriptor.size(0)<args.num_coarse_groups: raise ValueError("训练样本少于 coarse group 数量。")
    fit_data=_sample_numpy(descriptor,args.max_coarse_kmeans_samples,args.seed)
    km=MiniBatchKMeans(n_clusters=args.num_coarse_groups,init="k-means++",n_init=10,max_iter=args.kmeans_max_iter,batch_size=min(args.kmeans_batch_size,fit_data.shape[0]),random_state=args.seed,reassignment_ratio=0.01)
    km.fit(fit_data); labels=[]
    for s in range(0,descriptor.size(0),args.predict_chunk): labels.append(torch.from_numpy(km.predict(descriptor[s:s+args.predict_chunk].numpy())).long())
    labels=torch.cat(labels); counts=torch.bincount(labels,minlength=args.num_coarse_groups)
    print(f"Coarse KMeans 完成：groups={args.num_coarse_groups}, fit_samples={fit_data.shape[0]:,}, inertia={float(km.inertia_):.6f}")
    print("Coarse counts:",counts.tolist())
    return {"centers":torch.from_numpy(km.cluster_centers_).float(),"labels":labels,"counts":counts,"fit_sample_count":int(fit_data.shape[0]),"inertia":float(km.inertia_),"n_iter":int(getattr(km,"n_iter_",-1)),"n_steps":int(getattr(km,"n_steps_",-1))}

def summarize_coarse_latent(
    latents: torch.Tensor,
    coarse_labels: torch.Tensor,
    num_groups: int,
    count_power: float,
    variance_power: float,
) -> Dict[str, torch.Tensor]:
    counts = torch.bincount(coarse_labels, minlength=num_groups).long()
    trace = torch.zeros(num_groups, dtype=torch.float64)
    for group in range(num_groups):
        values = latents[coarse_labels == group].double()
        if values.size(0) > 1:
            trace[group] = values.var(dim=0, unbiased=False).sum()
    complexity = counts.double().clamp_min(1).pow(count_power)
    complexity = complexity * trace.clamp_min(1e-12).pow(variance_power)
    return {
        "counts": counts,
        "latent_trace": trace.float(),
        "complexity": complexity.float(),
    }


def _predict_two_centers(
    values: torch.Tensor,
    centers: torch.Tensor,
    chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = []
    squared_distances = []
    for start in range(0, values.size(0), chunk):
        distance_squared = (
            values[start:start + chunk, None, :] - centers[None, :, :]
        ).pow(2).sum(dim=-1)
        minimum, index = distance_squared.min(dim=1)
        labels.append(index.to(torch.int8))
        squared_distances.append(minimum)
    return torch.cat(labels), torch.cat(squared_distances)


def _fit_two_means_proposal(
    latents: torch.Tensor,
    member_indices: torch.Tensor,
    parent_center: torch.Tensor,
    seed: int,
    max_fit_samples: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
    predict_chunk: int,
    min_child_size: int,
) -> Dict[str, Any] | None:
    if member_indices.numel() < max(2, 2 * min_child_size):
        return None
    values = latents[member_indices]
    fit_data = _sample_numpy(values, max_fit_samples, seed)
    if fit_data.shape[0] < 2:
        return None
    kmeans = MiniBatchKMeans(
        n_clusters=2,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=min(batch_size, fit_data.shape[0]),
        random_state=seed,
        reassignment_ratio=0.0,
    )
    kmeans.fit(fit_data)
    child_centers = torch.from_numpy(kmeans.cluster_centers_).float()
    child_labels, child_squared_distance = _predict_two_centers(
        values, child_centers, predict_chunk,
    )
    child_count = torch.bincount(child_labels.long(), minlength=2)
    if int(child_count.min()) < min_child_size:
        return None
    parent_sse = float((values - parent_center).pow(2).sum())
    child_sse = float(child_squared_distance.sum())
    gain = parent_sse - child_sse
    if not math.isfinite(gain):
        return None
    return {
        "gain": gain,
        "parent_sse": parent_sse,
        "child_sse": child_sse,
        "child_centers": child_centers,
        "child_labels": child_labels,
        "child_count": child_count,
        "fit_sample_count": int(fit_data.shape[0]),
    }


def fit_local_greedy_split_gain(
    latents: torch.Tensor,
    coarse_labels: torch.Tensor,
    args,
) -> Dict[str, Any]:
    num_groups = int(args.num_coarse_groups)
    target_prototypes = int(args.num_prototypes)
    initial_per_group = int(args.min_prototypes_per_coarse)
    if target_prototypes < num_groups * initial_per_group:
        raise ValueError(
            "num_prototypes 小于 num_coarse_groups × min_prototypes_per_coarse。"
        )

    active: Dict[int, Dict[str, Any]] = {}
    next_cluster_id = 0
    initial_fit_counts = []
    initial_inertia = []

    for group in range(num_groups):
        group_indices = torch.nonzero(
            coarse_labels == group, as_tuple=False,
        ).flatten()
        values = latents[group_indices]
        if values.size(0) < initial_per_group:
            raise ValueError(
                f"coarse group {group} 样本数 {values.size(0)} "
                f"小于初始prototype数 {initial_per_group}。"
            )
        fit_data = _sample_numpy(
            values,
            args.max_local_kmeans_samples_per_group,
            args.seed + 1009 * group,
        )
        kmeans = MiniBatchKMeans(
            n_clusters=initial_per_group,
            init="k-means++",
            n_init=10,
            max_iter=args.kmeans_max_iter,
            batch_size=min(args.kmeans_batch_size, fit_data.shape[0]),
            random_state=args.seed + group,
            reassignment_ratio=0.01,
        )
        kmeans.fit(fit_data)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        local_labels = []
        for start in range(0, values.size(0), args.predict_chunk):
            local_labels.append(
                torch.from_numpy(
                    kmeans.predict(
                        values[start:start + args.predict_chunk].numpy()
                    )
                ).long()
            )
        local_labels = torch.cat(local_labels)
        initial_fit_counts.append(int(fit_data.shape[0]))
        initial_inertia.append(float(kmeans.inertia_))
        for local_id in range(initial_per_group):
            members = group_indices[local_labels == local_id]
            if members.numel() == 0:
                raise RuntimeError(
                    f"初始局部KMeans产生空cluster：group={group}, local={local_id}"
                )
            active[next_cluster_id] = {
                "group": group,
                "center": centers[local_id].clone(),
                "members": members,
            }
            next_cluster_id += 1
        print(
            f"Initial local KMeans group={group:02d}: N={values.size(0):,}, "
            f"K={initial_per_group}, fit={fit_data.shape[0]:,}, "
            f"inertia={float(kmeans.inertia_):.6f}"
        )

    proposals: Dict[int, Dict[str, Any] | None] = {}
    heap: List[Tuple[float, int]] = []

    def refresh_proposal(cluster_id: int, seed_offset: int) -> None:
        cluster = active[cluster_id]
        proposal = _fit_two_means_proposal(
            latents=latents,
            member_indices=cluster["members"],
            parent_center=cluster["center"],
            seed=args.seed + 200000 + seed_offset,
            max_fit_samples=args.budget_split_max_samples,
            batch_size=args.kmeans_batch_size,
            max_iter=args.budget_split_max_iter,
            n_init=args.budget_split_n_init,
            predict_chunk=args.predict_chunk,
            min_child_size=args.budget_split_min_cluster_size,
        )
        proposals[cluster_id] = proposal
        if proposal is not None:
            heapq.heappush(heap, (-float(proposal["gain"]), cluster_id))

    for cluster_id in list(active):
        refresh_proposal(cluster_id, cluster_id)

    split_log = []
    while len(active) < target_prototypes:
        chosen_id = None
        chosen_proposal = None
        while heap:
            negative_gain, cluster_id = heapq.heappop(heap)
            if cluster_id not in active:
                continue
            proposal = proposals.get(cluster_id)
            if proposal is None:
                continue
            if abs(-negative_gain - float(proposal["gain"])) > 1e-6 * max(
                1.0, abs(float(proposal["gain"]))
            ):
                continue
            chosen_id = cluster_id
            chosen_proposal = proposal
            break
        if chosen_id is None or chosen_proposal is None:
            raise RuntimeError(
                f"Greedy split-gain在{len(active)}个prototype时没有可拆分cluster，"
                f"无法达到目标{target_prototypes}。可减小"
                "--budget_split_min_cluster_size。"
            )
        if float(chosen_proposal["gain"]) <= args.budget_split_gain_floor:
            print(
                f"警告：剩余最佳split gain={float(chosen_proposal['gain']):.6f} "
                f"不高于floor={args.budget_split_gain_floor:.6f}，仍继续拆分以达到固定K。"
            )

        parent = active.pop(chosen_id)
        proposals.pop(chosen_id, None)
        child_assignment = chosen_proposal["child_labels"].long()
        child_ids = []
        for child_local in range(2):
            members = parent["members"][child_assignment == child_local]
            cluster_id = next_cluster_id
            next_cluster_id += 1
            active[cluster_id] = {
                "group": int(parent["group"]),
                "center": chosen_proposal["child_centers"][child_local].clone(),
                "members": members,
            }
            child_ids.append(cluster_id)

        split_log.append({
            "step": len(split_log),
            "parent_internal_id": int(chosen_id),
            "child_internal_ids": child_ids,
            "coarse_group": int(parent["group"]),
            "parent_count": int(parent["members"].numel()),
            "child_count": chosen_proposal["child_count"].tolist(),
            "parent_sse": float(chosen_proposal["parent_sse"]),
            "child_sse": float(chosen_proposal["child_sse"]),
            "split_gain": float(chosen_proposal["gain"]),
            "fit_sample_count": int(chosen_proposal["fit_sample_count"]),
        })

        for child_id in child_ids:
            refresh_proposal(child_id, next_cluster_id + child_id)

        if len(active) % 8 == 0 or len(active) == target_prototypes:
            budget_now = torch.bincount(
                torch.tensor(
                    [int(cluster["group"]) for cluster in active.values()],
                    dtype=torch.long,
                ),
                minlength=num_groups,
            )
            print(
                f"Greedy split-gain: K={len(active)}/{target_prototypes}, "
                f"last_gain={float(chosen_proposal['gain']):.6f}, "
                f"budget={budget_now.tolist()}"
            )

    ordered = sorted(
        active.items(),
        key=lambda item: (int(item[1]["group"]), int(item[0])),
    )
    centers = torch.stack([item[1]["center"] for item in ordered])
    center_groups = torch.tensor(
        [int(item[1]["group"]) for item in ordered], dtype=torch.long,
    )
    labels = torch.empty(latents.size(0), dtype=torch.long)
    cluster_sse = torch.zeros(len(ordered), dtype=torch.float64)
    for final_id, (_, cluster) in enumerate(ordered):
        members = cluster["members"]
        labels[members] = final_id
        cluster_sse[final_id] = (
            latents[members] - cluster["center"]
        ).pow(2).sum().double()

    budget = torch.bincount(center_groups, minlength=num_groups)
    group_inertia = torch.zeros(num_groups, dtype=torch.float64)
    group_inertia.index_add_(0, center_groups, cluster_sse)
    print("Greedy split-gain final budget:", budget.tolist(), "sum=", int(budget.sum()))
    for group in range(num_groups):
        count = int((coarse_labels == group).sum())
        rms = math.sqrt(float(group_inertia[group]) / max(count, 1))
        print(
            f"Final local group={group:02d}: N={count:,}, K={int(budget[group])}, "
            f"SSE={float(group_inertia[group]):.6f}, RMS={rms:.6f}"
        )
    return {
        "centers": centers,
        "center_coarse_ids": center_groups,
        "labels": labels,
        "budget": budget,
        "inertia": group_inertia.float(),
        "fit_counts": torch.tensor(initial_fit_counts),
        "initial_inertia": torch.tensor(initial_inertia),
        "split_gain_log": split_log,
        "budget_method": "greedy_bisecting_split_gain",
    }

def allocate_local_prototype_budget(latents:torch.Tensor,coarse_labels:torch.Tensor,num_groups:int,num_prototypes:int,min_per_group:int,count_power:float,variance_power:float)->Dict[str,torch.Tensor]:
    if num_prototypes<num_groups*min_per_group: raise ValueError("num_prototypes 小于 num_coarse_groups × min_prototypes_per_coarse。")
    counts=torch.bincount(coarse_labels,minlength=num_groups).double(); trace=torch.zeros(num_groups,dtype=torch.float64)
    for g in range(num_groups):
        x=latents[coarse_labels==g].double()
        if x.size(0)>1: trace[g]=x.var(dim=0,unbiased=False).sum()
    complexity=counts.clamp_min(1).pow(count_power)*trace.clamp_min(1e-12).pow(variance_power)
    budget=torch.full((num_groups,),min_per_group,dtype=torch.long); remaining=num_prototypes-int(budget.sum())
    if remaining>0:
        raw=complexity/complexity.sum().clamp_min(1e-12)*remaining; add=torch.floor(raw).long(); budget+=add; left=num_prototypes-int(budget.sum())
        if left>0:
            order=torch.argsort(raw-add.double(),descending=True); budget[order[:left]]+=1
    for g in range(num_groups):
        if budget[g]>counts[g]: raise ValueError(f"coarse group {g} 样本数 {int(counts[g])} 小于预算 {int(budget[g])}")
    print("Local prototype budget:",budget.tolist(),"sum=",int(budget.sum()))
    return {"budget":budget,"counts":counts.long(),"latent_trace":trace.float(),"complexity":complexity.float()}

def fit_local_kmeans(latents:torch.Tensor,coarse_labels:torch.Tensor,budget:torch.Tensor,args)->Dict[str,Any]:
    centers=[]; center_groups=[]; labels=torch.empty(latents.size(0),dtype=torch.long); offset=0; fit_counts=[]; inertia=[]
    for g in range(budget.numel()):
        idx=torch.nonzero(coarse_labels==g,as_tuple=False).flatten(); x=latents[idx]; k=int(budget[g])
        fit_data=_sample_numpy(x,args.max_local_kmeans_samples_per_group,args.seed+1009*g)
        km=MiniBatchKMeans(n_clusters=k,init="k-means++",n_init=10,max_iter=args.kmeans_max_iter,batch_size=min(args.kmeans_batch_size,fit_data.shape[0]),random_state=args.seed+g,reassignment_ratio=0.01)
        km.fit(fit_data); local=[]
        for s in range(0,x.size(0),args.predict_chunk): local.append(torch.from_numpy(km.predict(x[s:s+args.predict_chunk].numpy())).long())
        local=torch.cat(local); labels[idx]=local+offset; c=torch.from_numpy(km.cluster_centers_).float(); centers.append(c); center_groups.append(torch.full((k,),g,dtype=torch.long)); fit_counts.append(int(fit_data.shape[0])); inertia.append(float(km.inertia_)); offset+=k
        print(f"Local KMeans group={g:02d}: N={x.size(0):,}, K={k}, fit={fit_data.shape[0]:,}, inertia={float(km.inertia_):.6f}")
    return {"centers":torch.cat(centers),"center_coarse_ids":torch.cat(center_groups),"labels":labels,"fit_counts":torch.tensor(fit_counts),"inertia":torch.tensor(inertia)}

def assign_within_coarse(latents:torch.Tensor,coarse_labels:torch.Tensor,centers:torch.Tensor,center_coarse_ids:torch.Tensor,chunk:int,topk:int=1)->Tuple[torch.Tensor,torch.Tensor]:
    labels=torch.empty(latents.size(0),dtype=torch.long); distances=torch.empty((latents.size(0),topk),dtype=torch.float32)
    for g in torch.unique(coarse_labels).tolist():
        idx=torch.nonzero(coarse_labels==g,as_tuple=False).flatten(); pidx=torch.nonzero(center_coarse_ids==g,as_tuple=False).flatten(); x=latents[idx]; k=min(topk,pidx.numel()); out_i=[]; out_d=[]
        for s in range(0,x.size(0),chunk):
            d=torch.cdist(x[s:s+chunk],centers[pidx]); vals,loc=torch.topk(d,k=k,largest=False,dim=1); out_i.append(pidx[loc]); out_d.append(vals)
        ids=torch.cat(out_i); ds=torch.cat(out_d)
        labels[idx]=ids[:,0]; distances[idx,:k]=ds
        if k<topk:
            distances[idx,k:]=float("inf")
    return labels,distances

def compute_cluster_stats(latents:torch.Tensor,labels:torch.Tensor,centers:torch.Tensor)->Dict[str,torch.Tensor]:
    k=centers.size(0); count=torch.bincount(labels,minlength=k); residual=latents-centers[labels]; sq=residual.pow(2).sum(dim=-1); sse=torch.zeros(k,dtype=torch.float64); sse.index_add_(0,labels,sq.double())
    radius=torch.sqrt((sse/count.clamp_min(1).double()).clamp_min(0)).float()
    return {"count":count,"sse":sse,"radius":radius,"residual":residual}

@torch.no_grad()
def decode_prototypes(model:QCNetFM,prototype_latents:torch.Tensor,device:torch.device,chunk_size:int)->torch.Tensor:
    outputs=[]
    for s in range(0,prototype_latents.size(0),chunk_size):
        traj=model.latent_decoder(prototype_latents[s:s+chunk_size].to(device))
        if isinstance(traj,(tuple,list)): traj=traj[0]
        if traj.ndim==4 and traj.size(1)==1: traj=traj[:,0]
        if traj.ndim!=3: raise RuntimeError(f"prototype 解码轨迹形状错误：{tuple(traj.shape)}")
        outputs.append(traj.float().cpu())
    return torch.cat(outputs)

def trajectory_pair_metric(traj_a:torch.Tensor,traj_b:torch.Tensor,scale:float,fde_weight:float)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    diff=(traj_a-traj_b)*scale; step=diff.norm(dim=-1); ade=step.mean(dim=-1); fde=step[:,-1]; return ade,fde,ade+fde_weight*fde

def compute_harmful_overlap(centers:torch.Tensor,center_groups:torch.Tensor,radius:torch.Tensor,prototype_traj:torch.Tensor,trajectory_scale:float,neighbors:int,fde_weight:float)->Dict[str,torch.Tensor]:
    k=centers.size(0); pair=torch.cdist(centers,centers); pair.fill_diagonal_(float("inf")); nn=torch.topk(pair,k=min(neighbors,max(k-1,1)),largest=False,dim=1).indices
    pairs=set()
    for i in range(k):
        for j in nn[i].tolist(): pairs.add((min(i,j),max(i,j)))
    pi=torch.tensor([p[0] for p in sorted(pairs)],dtype=torch.long); pj=torch.tensor([p[1] for p in sorted(pairs)],dtype=torch.long)
    if pi.numel()==0:
        empty=torch.empty(0); return {"pair_i":pi,"pair_j":pj,"center_l2":empty,"separation_ratio":empty,"traj_ade_m":empty,"traj_fde_m":empty,"traj_score_m":empty,"harmful_score":empty,"same_coarse":torch.empty(0,dtype=torch.bool)}
    center_l2=(centers[pi]-centers[pj]).norm(dim=-1); ratio=center_l2/(radius[pi]+radius[pj]+1e-8); ade,fde,traj_score=trajectory_pair_metric(prototype_traj[pi],prototype_traj[pj],trajectory_scale,fde_weight); harmful=torch.relu(1.0-ratio)*traj_score
    return {"pair_i":pi,"pair_j":pj,"center_l2":center_l2,"separation_ratio":ratio,"traj_ade_m":ade,"traj_fde_m":fde,"traj_score_m":traj_score,"harmful_score":harmful,"same_coarse":center_groups[pi].eq(center_groups[pj])}

def _overlap_scalar(diag:Dict[str,torch.Tensor])->float:
    score=diag["harmful_score"]
    return float(score.mean()) if score.numel()>0 else 0.0

def _refit_affected_groups(latents:torch.Tensor,coarse_labels:torch.Tensor,centers:torch.Tensor,center_groups:torch.Tensor,groups:Sequence[int],args,round_id:int)->torch.Tensor:
    result=centers.clone()
    for g in sorted(set(int(x) for x in groups)):
        idx=torch.nonzero(coarse_labels==g,as_tuple=False).flatten(); pidx=torch.nonzero(center_groups==g,as_tuple=False).flatten(); x=latents[idx]
        fit_data=_sample_numpy(x,args.split_merge_max_fit_samples,args.seed+50000+round_id*101+g)
        init=result[pidx].numpy(); km=MiniBatchKMeans(n_clusters=pidx.numel(),init=init,n_init=1,max_iter=args.split_merge_refine_iter,batch_size=min(args.kmeans_batch_size,fit_data.shape[0]),random_state=args.seed+round_id+g,reassignment_ratio=0.0)
        km.fit(fit_data); result[pidx]=torch.from_numpy(km.cluster_centers_).float()
    return result

def split_merge_refinement(
    model: QCNetFM,
    latents: torch.Tensor,
    coarse_labels: torch.Tensor,
    centers: torch.Tensor,
    center_groups: torch.Tensor,
    device: torch.device,
    args,
) -> Dict[str, Any]:
    logs = []
    labels, _ = assign_within_coarse(
        latents, coarse_labels, centers, center_groups, args.predict_chunk, 1,
    )
    for round_id in range(args.split_merge_rounds):
        stats = compute_cluster_stats(latents, labels, centers)
        trajectories = decode_prototypes(
            model, centers.unsqueeze(1), device, args.decode_chunk,
        )
        diagnostics = compute_harmful_overlap(
            centers, center_groups, stats["radius"], trajectories,
            float(model.trajectory_scale), args.overlap_neighbors,
            args.overlap_fde_weight,
        )
        current_sse = float(stats["sse"].sum())
        current_harmful = _overlap_scalar(diagnostics)
        counts = stats["count"]

        merge_candidates = []
        for group in torch.unique(center_groups).tolist():
            prototype_ids = torch.nonzero(
                center_groups == group, as_tuple=False,
            ).flatten()
            if prototype_ids.numel() <= args.min_prototypes_per_coarse:
                continue
            for left in range(prototype_ids.numel()):
                for right in range(left + 1, prototype_ids.numel()):
                    first = int(prototype_ids[left])
                    second = int(prototype_ids[right])
                    count_first = int(counts[first])
                    count_second = int(counts[second])
                    if count_first == 0 or count_second == 0:
                        continue
                    merge_cost = (
                        count_first * count_second / (count_first + count_second)
                    ) * float((centers[first] - centers[second]).pow(2).sum())
                    _, _, trajectory_score = trajectory_pair_metric(
                        trajectories[first:first + 1],
                        trajectories[second:second + 1],
                        float(model.trajectory_scale),
                        args.overlap_fde_weight,
                    )
                    trajectory_score = float(trajectory_score[0])
                    if (
                        args.merge_max_traj_score_m > 0
                        and trajectory_score > args.merge_max_traj_score_m
                    ):
                        continue
                    ranking_score = (
                        merge_cost
                        + args.split_merge_traj_weight * trajectory_score
                    )
                    merge_candidates.append({
                        "first": first,
                        "second": second,
                        "merge_cost": merge_cost,
                        "trajectory_score": trajectory_score,
                        "ranking_score": ranking_score,
                    })

        if not merge_candidates:
            print("Split-merge: 没有可用merge pair，停止。")
            break
        merge_candidates.sort(key=lambda item: item["ranking_score"])
        merge_candidates = merge_candidates[:args.split_merge_merge_trials]

        split_candidates = []
        split_order = torch.argsort(stats["sse"], descending=True).tolist()
        for split_index in split_order:
            if len(split_candidates) >= args.split_merge_split_trials:
                break
            if int(counts[split_index]) < 2 * args.split_merge_min_cluster_size:
                continue
            member_indices = torch.nonzero(
                labels == split_index, as_tuple=False,
            ).flatten()
            proposal = _fit_two_means_proposal(
                latents=latents,
                member_indices=member_indices,
                parent_center=centers[split_index],
                seed=args.seed + 700000 + round_id * 1000 + split_index,
                max_fit_samples=args.split_merge_max_fit_samples,
                batch_size=args.kmeans_batch_size,
                max_iter=args.split_merge_refine_iter,
                n_init=3,
                predict_chunk=args.predict_chunk,
                min_child_size=args.split_merge_min_cluster_size,
            )
            if proposal is None:
                continue
            split_candidates.append({
                "index": int(split_index),
                "member_indices": member_indices,
                "proposal": proposal,
            })

        if not split_candidates:
            print("Split-merge: 没有可用split cluster，停止。")
            break

        combinations = []
        for merge in merge_candidates:
            for split in split_candidates:
                split_index = split["index"]
                if split_index in (merge["first"], merge["second"]):
                    continue
                predicted_net_gain = (
                    float(split["proposal"]["gain"]) - merge["merge_cost"]
                )
                combinations.append({
                    "merge": merge,
                    "split": split,
                    "predicted_net_gain": predicted_net_gain,
                })
        combinations.sort(
            key=lambda item: item["predicted_net_gain"], reverse=True,
        )

        accepted = False
        trial_logs = []
        for trial_id, combination in enumerate(
            combinations[:args.split_merge_candidate_trials]
        ):
            merge = combination["merge"]
            split = combination["split"]
            first = merge["first"]
            second = merge["second"]
            split_index = split["index"]
            proposal = split["proposal"]

            merged_center = (
                centers[first] * counts[first]
                + centers[second] * counts[second]
            ) / max(int(counts[first] + counts[second]), 1)
            keep = [
                index for index in range(centers.size(0))
                if index not in (first, second, split_index)
            ]
            candidate_centers = torch.cat([
                centers[keep],
                merged_center[None],
                proposal["child_centers"],
            ], dim=0)
            candidate_groups = torch.cat([
                center_groups[keep],
                center_groups[first:first + 1],
                center_groups[split_index:split_index + 1].repeat(2),
            ], dim=0)
            affected_groups = [
                int(center_groups[first]),
                int(center_groups[split_index]),
            ]
            candidate_centers = _refit_affected_groups(
                latents, coarse_labels, candidate_centers, candidate_groups,
                affected_groups, args, round_id,
            )
            candidate_labels, _ = assign_within_coarse(
                latents, coarse_labels, candidate_centers, candidate_groups,
                args.predict_chunk, 1,
            )
            candidate_stats = compute_cluster_stats(
                latents, candidate_labels, candidate_centers,
            )
            candidate_sse = float(candidate_stats["sse"].sum())
            candidate_trajectories = decode_prototypes(
                model, candidate_centers.unsqueeze(1), device, args.decode_chunk,
            )
            candidate_diagnostics = compute_harmful_overlap(
                candidate_centers, candidate_groups,
                candidate_stats["radius"], candidate_trajectories,
                float(model.trajectory_scale), args.overlap_neighbors,
                args.overlap_fde_weight,
            )
            candidate_harmful = _overlap_scalar(candidate_diagnostics)

            quantization_ok = candidate_sse <= current_sse * (
                1.0 + args.split_merge_quantization_tolerance
            )
            overlap_better = candidate_harmful <= current_harmful * (
                1.0 - args.split_merge_min_overlap_improvement
            )
            overlap_not_worse = candidate_harmful <= (
                current_harmful * (1.0 + args.split_merge_overlap_tolerance)
                + 1e-12
            )
            quantization_better = candidate_sse < current_sse * (1.0 - 1e-5)
            accepted_candidate = (
                quantization_ok
                and overlap_not_worse
                and (overlap_better or quantization_better)
            )
            reason = []
            if not quantization_ok:
                reason.append("quantization_tolerance")
            if not overlap_not_worse:
                reason.append("overlap_tolerance")
            if not (overlap_better or quantization_better):
                reason.append("no_objective_improvement")
            trial_record = {
                "trial": trial_id,
                "merge": [first, second],
                "split": split_index,
                "merge_total_cost": float(merge["merge_cost"]),
                "merge_trajectory_score_m": float(merge["trajectory_score"]),
                "split_exact_gain": float(proposal["gain"]),
                "predicted_net_gain": float(
                    combination["predicted_net_gain"]
                ),
                "sse_before": current_sse,
                "sse_after": candidate_sse,
                "harmful_before": current_harmful,
                "harmful_after": candidate_harmful,
                "accepted": bool(accepted_candidate),
                "reject_reason": reason,
            }
            trial_logs.append(trial_record)

            if accepted_candidate:
                centers = candidate_centers
                center_groups = candidate_groups
                labels = candidate_labels
                logs.append({
                    "round": round_id,
                    "accepted": True,
                    "selected_trial": trial_record,
                    "trials": trial_logs,
                })
                accepted = True
                print(
                    f"Split-merge round={round_id}: ACCEPT "
                    f"merge=({first},{second}) cost={merge['merge_cost']:.3f}, "
                    f"split={split_index} gain={proposal['gain']:.3f}, "
                    f"net={combination['predicted_net_gain']:.3f}, "
                    f"SSE {current_sse:.3f}->{candidate_sse:.3f}, "
                    f"harmful {current_harmful:.6f}->{candidate_harmful:.6f}"
                )
                break
            print(
                f"Split-merge round={round_id} trial={trial_id}: REJECT "
                f"merge=({first},{second}) cost={merge['merge_cost']:.3f}, "
                f"split={split_index} gain={proposal['gain']:.3f}, "
                f"net={combination['predicted_net_gain']:.3f}, "
                f"SSE={candidate_sse:.3f}, harmful={candidate_harmful:.6f}, "
                f"reason={','.join(reason) or 'unknown'}"
            )

        if not accepted:
            logs.append({
                "round": round_id,
                "accepted": False,
                "sse_before": current_sse,
                "harmful_before": current_harmful,
                "trials": trial_logs,
            })
            print(
                f"Split-merge round={round_id}: "
                "没有候选通过接受条件，停止。"
            )
            break

    labels, _ = assign_within_coarse(
        latents, coarse_labels, centers, center_groups, args.predict_chunk, 1,
    )
    return {
        "centers": centers,
        "center_coarse_ids": center_groups,
        "labels": labels,
        "log": logs,
    }


def select_medoids(latents:torch.Tensor,labels:torch.Tensor,centers:torch.Tensor,center_groups:torch.Tensor,coarse_labels:torch.Tensor)->Dict[str,torch.Tensor]:
    k=centers.size(0); medoid_idx=torch.empty(k,dtype=torch.long)
    for cid in range(k):
        member=torch.nonzero(labels==cid,as_tuple=False).flatten()
        if member.numel()==0:
            candidate=torch.nonzero(coarse_labels==center_groups[cid],as_tuple=False).flatten(); d=(latents[candidate]-centers[cid]).pow(2).sum(dim=-1); medoid_idx[cid]=candidate[d.argmin()]
        else:
            d=(latents[member]-centers[cid]).pow(2).sum(dim=-1); medoid_idx[cid]=member[d.argmin()]
    prototypes=latents[medoid_idx].contiguous(); return {"prototype_latents_centered_raw":prototypes.unsqueeze(1),"kmeans_centers_centered_raw":centers,"kmeans_medoid_source_flat_index":medoid_idx,"kmeans_medoid_to_center_raw_l2":(prototypes-centers).norm(dim=-1)}

def nearest_coarse_from_descriptor(descriptor:torch.Tensor,coarse_centers:torch.Tensor,chunk:int)->torch.Tensor:
    out=[]
    for s in range(0,descriptor.size(0),chunk): out.append(torch.cdist(descriptor[s:s+chunk],coarse_centers).argmin(dim=1))
    return torch.cat(out)

def assign_hierarchical_latent(
    z_gt: torch.Tensor,
    coarse_idx: torch.Tensor,
    prototype_latents: torch.Tensor,
    prototype_coarse_ids: torch.Tensor,
    chunk: int,
    support_margin_threshold: float,
    support_temperature: Any,
) -> Dict[str, torch.Tensor]:
    device = z_gt.device
    prototype = prototype_latents[:, 0].to(device)
    values = z_gt[:, 0]
    coarse_idx = coarse_idx.to(device)
    prototype_coarse_ids = prototype_coarse_ids.to(device)
    primary = torch.empty(values.size(0), dtype=torch.long, device=device)
    second = torch.full_like(primary, -1)
    distance1 = torch.empty(values.size(0), device=device)
    distance2 = torch.full_like(distance1, float("inf"))
    for group in torch.unique(coarse_idx).tolist():
        sample_indices = torch.nonzero(
            coarse_idx == group, as_tuple=False,
        ).flatten()
        prototype_indices = torch.nonzero(
            prototype_coarse_ids == group, as_tuple=False,
        ).flatten()
        group_values = values[sample_indices]
        topk = min(2, prototype_indices.numel())
        ids = []
        distances = []
        for start in range(0, group_values.size(0), chunk):
            current_distance, local_index = torch.topk(
                torch.cdist(
                    group_values[start:start + chunk],
                    prototype[prototype_indices],
                ),
                k=topk,
                largest=False,
                dim=1,
            )
            ids.append(prototype_indices[local_index])
            distances.append(current_distance)
        ids = torch.cat(ids)
        distances = torch.cat(distances)
        primary[sample_indices] = ids[:, 0]
        distance1[sample_indices] = distances[:, 0]
        if topk > 1:
            second[sample_indices] = ids[:, 1]
            distance2[sample_indices] = distances[:, 1]

    margin = (distance2 - distance1) / (distance1 + 1e-8)
    boundary = second.ge(0) & margin.lt(support_margin_threshold)
    if torch.is_tensor(support_temperature):
        temperature_table = support_temperature.to(
            device=device, dtype=distance1.dtype,
        ).flatten()
        sample_temperature = temperature_table[coarse_idx].clamp_min(1e-8)
    elif isinstance(support_temperature, (list, tuple, np.ndarray)):
        temperature_table = torch.as_tensor(
            support_temperature, device=device, dtype=distance1.dtype,
        ).flatten()
        sample_temperature = temperature_table[coarse_idx].clamp_min(1e-8)
    else:
        sample_temperature = torch.full_like(
            distance1, max(float(support_temperature), 1e-8),
        )
    logits = torch.stack([
        -distance1.pow(2) / sample_temperature,
        -distance2.pow(2) / sample_temperature,
    ], dim=1)
    weights = torch.softmax(logits, dim=1)
    weights[~boundary, 0] = 1.0
    weights[~boundary, 1] = 0.0
    support_ids = torch.stack([primary, second], dim=1)
    support_size = torch.where(
        boundary, torch.full_like(primary, 2), torch.ones_like(primary),
    )
    return {
        "primary": primary,
        "second": second,
        "distance1": distance1,
        "distance2": distance2,
        "margin": margin,
        "boundary": boundary,
        "support_ids": support_ids,
        "support_weights": weights,
        "support_size": support_size,
    }


def estimate_support_temperatures(
    latents: torch.Tensor,
    coarse_labels: torch.Tensor,
    prototype_latents: torch.Tensor,
    prototype_coarse_ids: torch.Tensor,
    args,
) -> Dict[str, Any]:
    num_groups = int(args.num_coarse_groups)
    if args.support_temperature > 0:
        per_coarse = torch.full(
            (num_groups,), float(args.support_temperature), dtype=torch.float32,
        )
        return {
            "global": float(args.support_temperature),
            "per_coarse": per_coarse,
            "count": torch.bincount(
                coarse_labels, minlength=num_groups,
            ),
        }

    result = assign_hierarchical_latent(
        latents.unsqueeze(1),
        coarse_labels,
        prototype_latents,
        prototype_coarse_ids,
        args.predict_chunk,
        args.support_margin_threshold,
        1.0,
    )
    squared_distance = result["distance1"].pow(2).float()
    global_temperature = float(
        squared_distance.median().clamp_min(
            args.support_temperature_floor
        )
    )
    per_coarse = torch.full(
        (num_groups,), global_temperature, dtype=torch.float32,
    )
    counts = torch.bincount(coarse_labels, minlength=num_groups)
    for group in range(num_groups):
        values = squared_distance[coarse_labels == group]
        if values.numel() >= args.support_temperature_min_samples:
            per_coarse[group] = values.median().clamp_min(
                args.support_temperature_floor
            )
    print(
        f"自动global support_temperature={global_temperature:.8f} "
        "（仅作为兼容字段）"
    )
    print(
        "Per-coarse support_temperature:",
        [round(float(value), 8) for value in per_coarse],
    )
    return {
        "global": global_temperature,
        "per_coarse": per_coarse,
        "count": counts,
    }


def save_json(path:Path,obj:Mapping[str,Any])->None:
    def convert(x):
        if isinstance(x,torch.Tensor): return x.tolist()
        if isinstance(x,Path): return str(x)
        if isinstance(x,np.generic): return x.item()
        raise TypeError
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert),encoding="utf-8")

def save_overlap_csv(path:Path,diag:Dict[str,torch.Tensor])->None:
    fields=["prototype_i","prototype_j","same_coarse","center_l2","separation_ratio","traj_ade_m","traj_fde_m","traj_score_m","harmful_score"]
    rows=[]
    for n in range(diag["pair_i"].numel()): rows.append({"prototype_i":int(diag["pair_i"][n]),"prototype_j":int(diag["pair_j"][n]),"same_coarse":bool(diag["same_coarse"][n]),"center_l2":float(diag["center_l2"][n]),"separation_ratio":float(diag["separation_ratio"][n]),"traj_ade_m":float(diag["traj_ade_m"][n]),"traj_fde_m":float(diag["traj_fde_m"][n]),"traj_score_m":float(diag["traj_score_m"][n]),"harmful_score":float(diag["harmful_score"][n])})
    rows.sort(key=lambda x:x["harmful_score"],reverse=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_vae_model(checkpoint_path:str,device:torch.device)->Tuple[QCNetFM,Dict[str,Any]]:
    checkpoint=load_checkpoint(checkpoint_path)
    cfg=dict(checkpoint.get("hyper_parameters",{}))
    if not cfg:
        raise RuntimeError("checkpoint 缺少 hyper_parameters。")
    for key,value in {"num_hist_tokens":2,"num_map_tokens":4,"num_agent_tokens":2,"vae_num_intents":1}.items():
        cfg.setdefault(key,value)
    cfg["residual_fm"]=False
    cfg["prototype_bank_path"]=None
    model=QCNetFM(**cfg)
    source_state=checkpoint.get("state_dict",checkpoint)
    current_state=model.state_dict()
    required_prefixes=("latent_encoder.","latent_decoder.")
    required_exact={"z_mean","z_std"}
    filtered={}
    for key,value in source_state.items():
        required=key.startswith(required_prefixes) or key in required_exact
        if required and torch.is_tensor(value) and key in current_state and tuple(value.shape)==tuple(current_state[key].shape):
            filtered[key]=value
    required_keys=[key for key in current_state if key.startswith(required_prefixes) or key in required_exact]
    missing=[key for key in required_keys if key not in filtered]
    if missing:
        raise RuntimeError(f"VAE 权重未完整加载，共缺失 {len(missing)} 项：\n"+"\n".join(missing[:40]))
    model.load_state_dict(filtered,strict=False)
    if int(model.vae_num_intents)!=1:
        raise ValueError(f"当前脚本要求 vae_num_intents=1，实际为 {model.vae_num_intents}。")
    model.to(device).eval()
    for module in (model.latent_encoder,model.latent_decoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    print(f"已加载 VAE tensor 数量：{len(filtered)}")
    print(f"latent_dim={model.latent_dim}, z_mean={model.z_mean.detach().cpu().tolist()}")
    return model,cfg


def build_vae_cache_datamodule(cfg:Dict[str,Any],args)->ArgoverseV2DataModule:
    dm_cfg=dict(cfg)
    dm_cfg.update({
        "root":args.root or dm_cfg.get("root") or ".",
        "train_batch_size":args.batch_size,
        "val_batch_size":args.batch_size,
        "test_batch_size":args.batch_size,
        "num_workers":args.num_workers,
        "pin_memory":args.pin_memory,
        "persistent_workers":args.persistent_workers and args.num_workers>0,
        "shuffle":False,
        "vae_processed_dir":args.vae_processed_dir,
    })
    if not dm_cfg.get("vae_processed_dir"):
        raise ValueError("缺少 --vae_processed_dir。")
    return ArgoverseV2DataModule(**dm_cfg)


def validate_vae_cache(datamodule:ArgoverseV2DataModule)->Path:
    cache_dir=Path(datamodule.vae_processed_dir).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"VAE 缓存目录不存在：{cache_dir}")
    files=[path for path in cache_dir.rglob("*") if path.is_file() and path.name!="_SUCCESS"]
    if not files:
        raise RuntimeError(f"VAE 缓存目录中没有可读取的数据文件：{cache_dir}")
    print(f"使用 VAE 未来轨迹缓存：{cache_dir}，文件数={len(files):,}")
    return cache_dir


@torch.no_grad()
def collect_train_latents_and_descriptors(model:QCNetFM,datamodule:ArgoverseV2DataModule,device:torch.device,args)->Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
    latent_chunks:List[torch.Tensor]=[]
    block_chunks:Dict[str,List[torch.Tensor]]={}
    loader=datamodule.vae_train_dataloader()
    for batch_idx,batch in enumerate(tqdm(loader,desc="Collect train latent + trajectory descriptors from VAE cache")):
        if args.max_latent_batches>0 and batch_idx>=args.max_latent_batches:
            break
        if not isinstance(batch,Mapping):
            raise TypeError(f"vae_train_dataloader 应返回 Mapping，实际为 {type(batch)}")
        target,predict_mask,valid=get_cached_targets(model,batch,device,args.agent_scope)
        if not valid.any():
            continue
        with autocast_context(device,args.precision):
            z=encode_centered_raw(model,target,predict_mask)
        blocks=build_trajectory_descriptor_blocks(
            target[valid],predict_mask[valid],float(model.trajectory_scale),
            args.dct_coeffs,args.num_segments,args.time_step,args.slow_speed_threshold,
        )
        latent_chunks.append(z[valid,0].cpu())
        for name,value in blocks.items():
            block_chunks.setdefault(name,[]).append(value.float().cpu())
    if not latent_chunks:
        raise RuntimeError("没有从 VAE 缓存收集到有效训练样本。")
    latents=torch.cat(latent_chunks).float()
    finite=torch.isfinite(latents).all(dim=-1)
    blocks={name:torch.cat(values).float()[finite] for name,values in block_chunks.items()}
    latents=latents[finite]
    print(f"训练样本：{latents.size(0):,}，latent_dim={latents.size(1)}，descriptor_dim={sum(x.size(1) for x in blocks.values())}")
    return latents,blocks


def parse_args():
    parser=argparse.ArgumentParser(description="仅使用 VAE 缓存建立 trajectory-aware hierarchical prototype bank。")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--vae_processed_dir",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--root",default=None,help="只用于兼容 DataModule；不会读取 train/val processed 数据。")
    parser.add_argument("--num_prototypes",type=int,default=128)
    parser.add_argument("--num_coarse_groups",type=int,default=16)
    parser.add_argument("--min_prototypes_per_coarse",type=int,default=2)
    parser.add_argument("--agent_scope",choices=["all_valid","focal"],default="all_valid")
    parser.add_argument("--max_latent_batches",type=int,default=0)
    parser.add_argument("--dct_coeffs",type=int,default=6)
    parser.add_argument("--num_segments",type=int,default=6)
    parser.add_argument("--time_step",type=float,default=0.1)
    parser.add_argument("--slow_speed_threshold",type=float,default=0.5)
    parser.add_argument("--descriptor_weight_dct",type=float,default=1.0)
    parser.add_argument("--descriptor_weight_displacement",type=float,default=1.0)
    parser.add_argument("--descriptor_weight_speed",type=float,default=0.5)
    parser.add_argument("--descriptor_weight_heading",type=float,default=1.0)
    parser.add_argument("--descriptor_weight_global",type=float,default=1.0)
    parser.add_argument("--budget_count_power",type=float,default=1.0)
    parser.add_argument("--budget_variance_power",type=float,default=0.5)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--pin_memory",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--persistent_workers",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--precision",choices=["32","bf16","fp16"],default="bf16")
    parser.add_argument("--seed",type=int,default=2030)
    parser.add_argument("--kmeans_batch_size",type=int,default=8192)
    parser.add_argument("--kmeans_max_iter",type=int,default=300)
    parser.add_argument("--max_coarse_kmeans_samples",type=int,default=1000000)
    parser.add_argument("--max_local_kmeans_samples_per_group",type=int,default=0)
    parser.add_argument("--local_budget_method",choices=["greedy_split_gain","formula"],default="greedy_split_gain")
    parser.add_argument("--budget_split_max_samples",type=int,default=200000)
    parser.add_argument("--budget_split_max_iter",type=int,default=60)
    parser.add_argument("--budget_split_n_init",type=int,default=3)
    parser.add_argument("--budget_split_min_cluster_size",type=int,default=1000)
    parser.add_argument("--budget_split_gain_floor",type=float,default=0.0)
    parser.add_argument("--predict_chunk",type=int,default=65536)
    parser.add_argument("--split_merge_rounds",type=int,default=3)
    parser.add_argument("--split_merge_eta",type=float,default=0.4)
    parser.add_argument("--split_merge_min_cluster_size",type=int,default=1000)
    parser.add_argument("--split_merge_max_cov_samples",type=int,default=200000)
    parser.add_argument("--split_merge_max_fit_samples",type=int,default=500000)
    parser.add_argument("--split_merge_refine_iter",type=int,default=50)
    parser.add_argument("--split_merge_candidate_trials",type=int,default=5)
    parser.add_argument("--split_merge_merge_trials",type=int,default=12)
    parser.add_argument("--split_merge_split_trials",type=int,default=8)
    parser.add_argument("--split_merge_quantization_tolerance",type=float,default=0.01)
    parser.add_argument("--split_merge_min_overlap_improvement",type=float,default=0.005)
    parser.add_argument("--split_merge_overlap_tolerance",type=float,default=0.01)
    parser.add_argument("--split_merge_traj_weight",type=float,default=0.05)
    parser.add_argument("--merge_max_traj_score_m",type=float,default=2.0)
    parser.add_argument("--overlap_neighbors",type=int,default=5)
    parser.add_argument("--overlap_fde_weight",type=float,default=0.5)
    parser.add_argument("--support_margin_threshold",type=float,default=0.15)
    parser.add_argument("--support_temperature",type=float,default=0.0)
    parser.add_argument("--support_temperature_floor",type=float,default=1e-6)
    parser.add_argument("--support_temperature_min_samples",type=int,default=100)
    parser.add_argument("--decode_chunk",type=int,default=256)
    parser.add_argument("--overwrite",action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"输出目录已存在且非空：{output_dir}。如需覆盖请加 --overwrite。"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("实际运行设备:", device)
    model, cfg = load_vae_model(args.checkpoint, device)
    datamodule = build_vae_cache_datamodule(cfg, args)
    cache_dir = validate_vae_cache(datamodule)
    train_latents, descriptor_blocks = collect_train_latents_and_descriptors(
        model, datamodule, device, args,
    )
    descriptor_stats = fit_descriptor_normalizer(
        descriptor_blocks, descriptor_block_weights(args),
    )
    train_descriptor = transform_descriptor_blocks(
        descriptor_blocks, descriptor_stats,
    )
    coarse = fit_coarse_kmeans(train_descriptor, args)
    coarse_summary = summarize_coarse_latent(
        train_latents,
        coarse["labels"],
        args.num_coarse_groups,
        args.budget_count_power,
        args.budget_variance_power,
    )

    if args.local_budget_method == "greedy_split_gain":
        local = fit_local_greedy_split_gain(
            train_latents, coarse["labels"], args,
        )
        budget_info = {
            "budget": local["budget"],
            "counts": coarse_summary["counts"],
            "latent_trace": coarse_summary["latent_trace"],
            "complexity": coarse_summary["complexity"],
            "method": local["budget_method"],
            "split_gain_log": local["split_gain_log"],
        }
    else:
        budget_info = allocate_local_prototype_budget(
            train_latents,
            coarse["labels"],
            args.num_coarse_groups,
            args.num_prototypes,
            args.min_prototypes_per_coarse,
            args.budget_count_power,
            args.budget_variance_power,
        )
        budget_info["method"] = "formula_count_variance"
        budget_info["split_gain_log"] = []
        local = fit_local_kmeans(
            train_latents, coarse["labels"], budget_info["budget"], args,
        )

    if args.split_merge_rounds > 0:
        refined = split_merge_refinement(
            model,
            train_latents,
            coarse["labels"],
            local["centers"],
            local["center_coarse_ids"],
            device,
            args,
        )
    else:
        refined = {
            "centers": local["centers"],
            "center_coarse_ids": local["center_coarse_ids"],
            "labels": local["labels"],
            "log": [],
        }

    # 先对split-merge后的连续中心做一次全量重分配，作为medoid替换前基线。
    center_labels, center_distances = assign_within_coarse(
        train_latents,
        coarse["labels"],
        refined["centers"],
        refined["center_coarse_ids"],
        args.predict_chunk,
        1,
    )
    center_stats = compute_cluster_stats(
        train_latents, center_labels, refined["centers"],
    )
    center_total_sse = float(center_stats["sse"].sum())
    center_rms_l2 = math.sqrt(
        center_total_sse / max(int(train_latents.size(0)), 1)
    )
    center_mean_l2 = float(center_distances[:, 0].mean())
    center_median_l2 = float(center_distances[:, 0].median())
    center_q90_l2 = float(
        torch.quantile(center_distances[:, 0], 0.90)
    )

    medoid = select_medoids(
        train_latents,
        center_labels,
        refined["centers"],
        refined["center_coarse_ids"],
        coarse["labels"],
    )
    prototype_latents = medoid["prototype_latents_centered_raw"]
    prototype_coarse_ids = refined["center_coarse_ids"]

    # 用最终保存的medoid重新分配全部训练latent，统计真实最终能量。
    final_labels, final_distances = assign_within_coarse(
        train_latents,
        coarse["labels"],
        prototype_latents[:, 0],
        prototype_coarse_ids,
        args.predict_chunk,
        2,
    )
    final_stats = compute_cluster_stats(
        train_latents, final_labels, prototype_latents[:, 0],
    )
    medoid_total_sse = float(final_stats["sse"].sum())
    medoid_rms_l2 = math.sqrt(
        medoid_total_sse / max(int(train_latents.size(0)), 1)
    )
    medoid_mean_l2 = float(final_distances[:, 0].mean())
    medoid_median_l2 = float(final_distances[:, 0].median())
    medoid_q90_l2 = float(
        torch.quantile(final_distances[:, 0], 0.90)
    )
    medoid_sse_increase = medoid_total_sse - center_total_sse
    medoid_sse_increase_ratio = (
        medoid_sse_increase / center_total_sse
        if center_total_sse > 0 else 0.0
    )

    print("\n========== Final prototype energy statistics ==========")
    print(
        f"Continuous centers: N={train_latents.size(0):,}, "
        f"SSE={center_total_sse:.6f}, RMS_L2={center_rms_l2:.6f}, "
        f"mean_L2={center_mean_l2:.6f}, "
        f"median_L2={center_median_l2:.6f}, q90_L2={center_q90_l2:.6f}"
    )
    print(
        f"Final medoids:      N={train_latents.size(0):,}, "
        f"SSE={medoid_total_sse:.6f}, RMS_L2={medoid_rms_l2:.6f}, "
        f"mean_L2={medoid_mean_l2:.6f}, "
        f"median_L2={medoid_median_l2:.6f}, q90_L2={medoid_q90_l2:.6f}"
    )
    print(
        f"Medoid replacement cost: ΔSSE={medoid_sse_increase:.6f}, "
        f"relative={medoid_sse_increase_ratio:.6%}, "
        f"mean medoid-to-center L2="
        f"{float(medoid['kmeans_medoid_to_center_raw_l2'].mean()):.6f}, "
        f"max={float(medoid['kmeans_medoid_to_center_raw_l2'].max()):.6f}"
    )
    print("=======================================================\n")
    prototype_trajectory = decode_prototypes(
        model, prototype_latents, device, args.decode_chunk,
    )
    overlap = compute_harmful_overlap(
        prototype_latents[:, 0],
        prototype_coarse_ids,
        final_stats["radius"],
        prototype_trajectory,
        float(model.trajectory_scale),
        args.overlap_neighbors,
        args.overlap_fde_weight,
    )
    support_temperature = estimate_support_temperatures(
        train_latents,
        coarse["labels"],
        prototype_latents,
        prototype_coarse_ids,
        args,
    )
    final_budget = torch.bincount(
        prototype_coarse_ids, minlength=args.num_coarse_groups,
    )
    bank = {
        **medoid,
        "prototype_latents_raw":
            prototype_latents + model.z_mean.detach().cpu().float(),
        "prototype_trajectories_normalized": prototype_trajectory,
        "prototype_trajectories_m":
            prototype_trajectory * float(model.trajectory_scale),
        "prototype_coarse_ids": prototype_coarse_ids,
        "prototype_train_count_initial": torch.bincount(
            final_labels, minlength=args.num_prototypes,
        ),
        "prototype_train_radius_raw_l2_initial": final_stats["radius"],
        "continuous_center_train_energy": {
            "num_samples": int(train_latents.size(0)),
            "sse_raw_l2": center_total_sse,
            "rms_raw_l2": center_rms_l2,
            "mean_raw_l2": center_mean_l2,
            "median_raw_l2": center_median_l2,
            "q90_raw_l2": center_q90_l2,
        },
        "final_medoid_train_energy": {
            "num_samples": int(train_latents.size(0)),
            "sse_raw_l2": medoid_total_sse,
            "rms_raw_l2": medoid_rms_l2,
            "mean_raw_l2": medoid_mean_l2,
            "median_raw_l2": medoid_median_l2,
            "q90_raw_l2": medoid_q90_l2,
            "sse_increase_vs_center": medoid_sse_increase,
            "sse_increase_ratio_vs_center": medoid_sse_increase_ratio,
            "mean_medoid_to_center_raw_l2":
                float(medoid["kmeans_medoid_to_center_raw_l2"].mean()),
            "max_medoid_to_center_raw_l2":
                float(medoid["kmeans_medoid_to_center_raw_l2"].max()),
        },
        "coarse_kmeans_centers_descriptor": coarse["centers"],
        "coarse_train_count": coarse["counts"],
        "coarse_local_prototype_budget_initial": budget_info["budget"],
        "coarse_local_prototype_budget_final": final_budget,
        "coarse_latent_trace": budget_info["latent_trace"],
        "coarse_budget_complexity": budget_info["complexity"],
        "local_budget_method": budget_info["method"],
        "local_budget_split_gain_log": budget_info["split_gain_log"],
        "descriptor_normalizer": descriptor_stats,
        "descriptor_config": {
            "dct_coeffs": args.dct_coeffs,
            "num_segments": args.num_segments,
            "time_step": args.time_step,
            "slow_speed_threshold": args.slow_speed_threshold,
            "block_weights": descriptor_block_weights(args),
        },
        "split_merge_log": refined["log"],
        "support_margin_threshold": args.support_margin_threshold,
        "support_temperature": support_temperature["global"],
        "support_temperature_per_coarse":
            support_temperature["per_coarse"],
        "support_temperature_count":
            support_temperature["count"],
        "overlap_diagnostics": overlap,
        "z_mean": model.z_mean.detach().cpu().float(),
        "z_std": model.z_std.detach().cpu().float(),
        "latent_dim": int(model.latent_dim),
        "num_intents": int(model.vae_num_intents),
        "trajectory_scale": float(model.trajectory_scale),
        "num_prototypes": int(args.num_prototypes),
        "num_coarse_groups": int(args.num_coarse_groups),
        "agent_scope": args.agent_scope,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "prototype_assignment_rule":
            "trajectory_descriptor_coarse_then_nearest_raw_centered_latent_l2",
        "kmeans_space":
            "trajectory_descriptor_coarse_plus_raw_centered_latent_local",
        "prototype_latent_statistic": "posterior_mean_mu",
        "train_latent_source": "vae_future_trajectory_cache",
        "vae_processed_dir": str(cache_dir),
        "refinement_applied": any(
            bool(item.get("accepted", False)) for item in refined["log"]
        ),
        "trajectory_metrics_used_for_coarse_partition": True,
        "trajectory_metrics_used_for_leaf_assignment": False,
        "assignment_statistics_attached": False,
        "format_version": 6,
    }
    torch.save(bank, output_dir / "prototype_bank.pt")
    save_overlap_csv(output_dir / "harmful_overlap_pairs.csv", overlap)
    manifest = {
        "format_version": 6,
        "mode": "bank_only",
        "checkpoint": bank["checkpoint"],
        "vae_processed_dir": str(cache_dir),
        "num_train_agents": int(train_latents.size(0)),
        "num_prototypes": int(args.num_prototypes),
        "num_coarse_groups": int(args.num_coarse_groups),
        "local_budget_method": budget_info["method"],
        "coarse_local_prototype_budget_final": final_budget,
        "continuous_center_train_energy": {
            "sse_raw_l2": center_total_sse,
            "rms_raw_l2": center_rms_l2,
            "mean_raw_l2": center_mean_l2,
            "median_raw_l2": center_median_l2,
            "q90_raw_l2": center_q90_l2,
        },
        "final_medoid_train_energy": {
            "sse_raw_l2": medoid_total_sse,
            "rms_raw_l2": medoid_rms_l2,
            "mean_raw_l2": medoid_mean_l2,
            "median_raw_l2": medoid_median_l2,
            "q90_raw_l2": medoid_q90_l2,
            "sse_increase_vs_center": medoid_sse_increase,
            "sse_increase_ratio_vs_center": medoid_sse_increase_ratio,
        },
        "support_temperature": support_temperature["global"],
        "support_temperature_per_coarse":
            support_temperature["per_coarse"],
        "refinement_applied": bool(bank["refinement_applied"]),
        "prototype_bank": str(
            (output_dir / "prototype_bank.pt").resolve()
        ),
    }
    save_json(output_dir / "bank_manifest.json", manifest)
    print(f"\n完成。Prototype 库：{output_dir / 'prototype_bank.pt'}")
    print(f"有害重叠诊断：{output_dir / 'harmful_overlap_pairs.csv'}")
    print(
        "该脚本没有读取train_processed_dir/val_processed_dir，"
        "也没有写入数据集。"
    )



if __name__=="__main__":
    main()
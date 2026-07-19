# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, ast, contextlib, csv, inspect, json, shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
import torch
from tqdm import tqdm
from datamodules import ArgoverseV2DataModule
from datasets import ArgoverseV2Dataset
from build_latent_prototypes_and_assign import (
    autocast_context,
    build_trajectory_descriptor_blocks,
    encode_centered_raw,
    load_vae_model,
    nearest_coarse_from_descriptor,
    resolve_device,
    set_seed,
    transform_descriptor_blocks,
)


TOP2_ASSIGNMENT_FIELDS=(
    "coarse_index","prototype_index","secondary_prototype_index",
    "support_prototype_ids","support_weights","support_size",
    "boundary_margin","is_boundary","valid_agent_mask",
    "match_latent_raw_l2","second_match_latent_raw_l2",
    "match_ade_m","match_fde_m","match_traj_score_m",
    "z_gt_centered_raw","z_residual",
)

def patch_dataset_assignment_fields(enabled:bool)->str|None:
    if not enabled: return None
    source=inspect.getsourcefile(ArgoverseV2Dataset)
    if not source: raise RuntimeError("无法定位 ArgoverseV2Dataset 源文件。")
    path=Path(source).resolve(); text=path.read_text(encoding="utf-8")
    if all(f'"{name}"' in text or f"'{name}'" in text for name in TOP2_ASSIGNMENT_FIELDS):
        print(f"Dataset已包含Top-2 assignment字段：{path}"); return str(path)
    tree=ast.parse(text); method=None
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=="ArgoverseV2Dataset":
            method=next((item for item in node.body if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name=="_merge_prototype_assignment"),None)
            break
    if method is None: raise RuntimeError(f"{path} 中未找到 ArgoverseV2Dataset._merge_prototype_assignment。")
    target=None
    for node in ast.walk(method):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="fields" for t in node.targets) and isinstance(node.value,(ast.Tuple,ast.List)):
            target=node; break
    if target is None: raise RuntimeError(f"{path} 中未找到 _merge_prototype_assignment 的 fields 元组。")
    indent=" "*(target.col_offset); replacement=indent+"fields=(\n"+indent+"    "+",\n".join(f'"{name}"' for name in TOP2_ASSIGNMENT_FIELDS)+",\n"+indent+")"
    lines=text.splitlines(); patched="\n".join(lines[:target.lineno-1]+replacement.splitlines()+lines[target.end_lineno:])+"\n"
    ast.parse(patched)
    backup=path.with_suffix(path.suffix+".bak_before_top2_sidecar")
    if not backup.exists(): backup.write_text(text,encoding="utf-8")
    path.write_text(patched,encoding="utf-8")
    print(f"已自动扩展Dataset的Top-2 assignment字段：{path}"); print(f"备份：{backup}")
    return str(path)


def normalize_scenario_id(value:Any)->str:
    if isinstance(value,str): return value
    if isinstance(value,bytes): return value.decode("utf-8")
    if isinstance(value,torch.Tensor):
        if value.numel()==1: return str(value.detach().cpu().item())
        return str(value.detach().cpu().tolist())
    return str(value)


def get_loader(datamodule:ArgoverseV2DataModule,split:str):
    if split=="train": return datamodule.train_dataloader()
    if split=="val": return datamodule.val_dataloader()
    raise ValueError(f"不支持 split={split}")


def get_targets(model,data,agent_scope:str):
    target=data["agent"]["target"][...,:model.output_dim]/float(model.trajectory_scale)
    predict_mask=data["agent"]["predict_mask"][:,model.num_historical_steps:].bool()
    valid=predict_mask.any(dim=-1)
    category=data["agent"]["category"].long() if "category" in data["agent"] else torch.full(
        (target.size(0),),-1,device=target.device,dtype=torch.long,
    )
    if agent_scope=="focal": valid=valid&category.eq(3)
    return target,predict_mask,valid,category


def get_agent_ptr(data)->torch.Tensor:
    store=data["agent"]
    n=int(store["target"].size(0))
    if "ptr" in store and store["ptr"] is not None:
        return store["ptr"].detach().cpu().long()
    return torch.tensor([0,n],dtype=torch.long)


def normalize_scenario_ids(value:Any,count:int)->List[str]:
    if isinstance(value,str): return [value]
    if isinstance(value,torch.Tensor):
        values=value.detach().cpu().tolist()
        values=values if isinstance(values,list) else [values]
        return [normalize_scenario_id(item) for item in values]
    if isinstance(value,(list,tuple)):
        return [normalize_scenario_id(item) for item in value]
    return [normalize_scenario_id(value)] if count==1 else [f"unknown_{i}" for i in range(count)]


def get_scenario_ids(data,count:int)->List[str]:
    for key in ("scenario_id","scenario_ids"):
        try:
            if key in data:
                ids=normalize_scenario_ids(data[key],count)
                if len(ids)==count: return ids
        except Exception:
            pass
    raise RuntimeError("无法从 batch 中读取 scenario_id。")




def build_datamodule(hparams:Dict[str,Any],args)->ArgoverseV2DataModule:
    cfg=dict(hparams)
    overrides={
        "root":args.root,
        "train_batch_size":args.batch_size,
        "val_batch_size":args.batch_size,
        "test_batch_size":args.batch_size,
        "num_workers":args.num_workers,
        "pin_memory":args.pin_memory,
        "persistent_workers":args.persistent_workers and args.num_workers>0,
        "shuffle":False,
        "train_raw_dir":args.train_raw_dir,
        "val_raw_dir":args.val_raw_dir,
        "train_processed_dir":args.train_processed_dir,
        "val_processed_dir":args.val_processed_dir,
    }
    for key,value in overrides.items():
        if value is not None: cfg[key]=value
    cfg.setdefault("dataset","argoverse_v2")
    if not cfg.get("root"): raise ValueError("缺少 --root。")
    datamodule=ArgoverseV2DataModule(**cfg)
    if "train" in args.splits:
        processed=args.train_processed_dir or cfg.get("train_processed_dir") or str(Path(cfg["root"])/"train"/"processed")
        datamodule.train_processed_dir=processed
        datamodule.train_dataset=ArgoverseV2Dataset(
            datamodule.root,"train",datamodule.train_raw_dir,processed,datamodule.train_transform,
        )
    if "val" in args.splits:
        processed=args.val_processed_dir or cfg.get("val_processed_dir") or str(Path(cfg["root"])/"val"/"processed")
        datamodule.val_processed_dir=processed
        datamodule.val_dataset=ArgoverseV2Dataset(
            datamodule.root,"val",datamodule.val_raw_dir,processed,datamodule.val_transform,
        )
    return datamodule


def assign_global(z_gt:torch.Tensor,prototype_latents:torch.Tensor,chunk:int,margin_threshold:float,temperature:Any)->Dict[str,torch.Tensor]:
    values=z_gt[:,0]; prototype=prototype_latents[:,0].to(values.device); ids=[]; distances=[]
    topk=min(2,prototype.size(0))
    for start in range(0,values.size(0),chunk):
        d=torch.cdist(values[start:start+chunk],prototype); current_distance,current_id=torch.topk(d,k=topk,largest=False,dim=1)
        ids.append(current_id); distances.append(current_distance)
    ids=torch.cat(ids); distances=torch.cat(distances); primary=ids[:,0]; distance1=distances[:,0]
    if topk>1: second=ids[:,1]; distance2=distances[:,1]
    else: second=torch.full_like(primary,-1); distance2=torch.full_like(distance1,float("inf"))
    margin=(distance2-distance1)/(distance1+1e-8); boundary=second.ge(0)&margin.lt(margin_threshold)
    if torch.is_tensor(temperature): temp=float(temperature.flatten()[0])
    elif isinstance(temperature,(list,tuple,np.ndarray)): temp=float(np.asarray(temperature).reshape(-1)[0])
    else: temp=float(temperature)
    logits=torch.stack([-distance1.square()/max(temp,1e-8),-distance2.square()/max(temp,1e-8)],dim=1)
    weights=torch.softmax(logits,dim=1); weights[~boundary,0]=1.0; weights[~boundary,1]=0.0
    support_ids=torch.stack([primary,second],dim=1); support_size=torch.where(boundary,torch.full_like(primary,2),torch.ones_like(primary))
    return {"primary":primary,"second":second,"distance1":distance1,"distance2":distance2,"margin":margin,"boundary":boundary,"support_ids":support_ids,"support_weights":weights,"support_size":support_size}

def assign_hierarchical(
    z_gt: torch.Tensor,
    coarse_idx: torch.Tensor,
    prototype_latents: torch.Tensor,
    prototype_coarse_ids: torch.Tensor,
    chunk: int,
    margin_threshold: float,
    temperature: Any,
) -> Dict[str, torch.Tensor]:
    device = z_gt.device
    prototype = prototype_latents[:, 0].to(device)
    groups = prototype_coarse_ids.to(device)
    values = z_gt[:, 0]
    coarse_idx = coarse_idx.to(device)
    primary = torch.empty(values.size(0), dtype=torch.long, device=device)
    second = torch.full_like(primary, -1)
    distance1 = torch.empty(values.size(0), device=device)
    distance2 = torch.full_like(distance1, float("inf"))
    for group in torch.unique(coarse_idx).tolist():
        sample_indices = torch.nonzero(
            coarse_idx == group, as_tuple=False,
        ).flatten()
        prototype_indices = torch.nonzero(
            groups == group, as_tuple=False,
        ).flatten()
        if prototype_indices.numel() == 0:
            raise RuntimeError(
                f"coarse group={group} 没有leaf prototype。"
            )
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
    boundary = second.ge(0) & margin.lt(margin_threshold)
    if torch.is_tensor(temperature):
        table = temperature.to(
            device=device, dtype=distance1.dtype,
        ).flatten()
        sample_temperature = table[coarse_idx].clamp_min(1e-8)
    elif isinstance(temperature, (list, tuple, np.ndarray)):
        table = torch.as_tensor(
            temperature, device=device, dtype=distance1.dtype,
        ).flatten()
        sample_temperature = table[coarse_idx].clamp_min(1e-8)
    else:
        sample_temperature = torch.full_like(
            distance1, max(float(temperature), 1e-8),
        )
    logits = torch.stack([
        -distance1.square() / sample_temperature,
        -distance2.square() / sample_temperature,
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



def trajectory_metrics(
    target_normalized:torch.Tensor,
    predict_mask:torch.Tensor,
    selected_trajectory_normalized:torch.Tensor,
    trajectory_scale:float,
)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    diff=(target_normalized-selected_trajectory_normalized)*float(trajectory_scale)
    distance=diff.norm(dim=-1)
    valid_count=predict_mask.sum(dim=-1).clamp_min(1).float()
    ade=(distance*predict_mask.float()).sum(dim=-1)/valid_count
    time_ids=torch.arange(target_normalized.size(1),device=target_normalized.device).view(1,-1)
    last=time_ids.masked_fill(~predict_mask,-1).max(dim=1).values.clamp_min(0)
    fde=distance.gather(1,last[:,None]).squeeze(1)
    return ade,fde,ade+0.5*fde


class ResidualStats:
    def __init__(self,num_prototypes:int,latent_dim:int):
        self.count=torch.zeros(num_prototypes,dtype=torch.long)
        self.sum=torch.zeros(num_prototypes,latent_dim,dtype=torch.float64)
        self.square_sum=torch.zeros(num_prototypes,latent_dim,dtype=torch.float64)

    def update(self,index:torch.Tensor,residual:torch.Tensor)->None:
        index=index.detach().cpu().long()
        residual=residual.detach().cpu().double()
        valid=index.ge(0)&index.lt(self.count.numel())&torch.isfinite(residual).all(dim=-1)
        if not valid.any(): return
        index=index[valid]; residual=residual[valid]
        self.count+=torch.bincount(index,minlength=self.count.numel())
        self.sum.index_add_(0,index,residual)
        self.square_sum.index_add_(0,index,residual.square())

    def finalize(self)->Dict[str,torch.Tensor]:
        count=self.count
        mean=torch.zeros_like(self.sum)
        var=torch.zeros_like(self.sum)
        supported=count.gt(0)
        mean[supported]=self.sum[supported]/count[supported,None]
        var[supported]=(self.square_sum[supported]/count[supported,None]-mean[supported].square()).clamp_min(0)
        total=int(count.sum())
        global_sum=self.sum.sum(dim=0)
        global_square=self.square_sum.sum(dim=0)
        global_mean=global_sum/max(total,1)
        global_var=(global_square/max(total,1)-global_mean.square()).clamp_min(0)
        return {
            "prototype_residual_count":count,
            "prototype_residual_mean":mean.float().unsqueeze(1),
            "prototype_residual_var_population":var.float().unsqueeze(1),
            "prototype_residual_std_population":var.sqrt().float().unsqueeze(1),
            "prototype_residual_supported_mask":supported,
            "empty_prototype_indices":torch.nonzero(~supported,as_tuple=False).flatten(),
            "residual_count":total,
            "residual_mean":global_mean.float(),
            "residual_var":global_var.float(),
            "residual_std":global_var.sqrt().float(),
            "residual_statistics_assignment":"bank_assignment_primary_prototype",
            "residual_statistics_normalization_applied":False,
        }


def save_residual_stats_csv(path:Path,stats:Dict[str,torch.Tensor])->None:
    count=stats["prototype_residual_count"]
    mean=stats["prototype_residual_mean"].squeeze(1)
    std=stats["prototype_residual_std_population"].squeeze(1)
    with path.open("w",newline="",encoding="utf-8") as handle:
        fields=["prototype","count"]+[item for dim in range(mean.size(1)) for item in (f"mean_d{dim}",f"std_d{dim}")]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for index in range(count.numel()):
            row={"prototype":index,"count":int(count[index])}
            for dim in range(mean.size(1)):
                row[f"mean_d{dim}"]=float(mean[index,dim])
                row[f"std_d{dim}"]=float(std[index,dim])
            writer.writerow(row)


def prepare_output_dir(path:Path,overwrite:bool)->None:
    if path.exists() and any(path.iterdir()):
        if overwrite: shutil.rmtree(path)
        else: raise FileExistsError(f"输出目录已存在且非空：{path}。如需覆盖请加 --overwrite。")
    path.mkdir(parents=True,exist_ok=True)


class AssignmentShardWriter:
    def __init__(self,output_dir:Path,split:str,shard_size:int,prototype_bank:str,overwrite:bool):
        self.output_dir=output_dir; self.split=split; self.shard_size=max(int(shard_size),1); self.prototype_bank=prototype_bank
        if output_dir.exists() and any(output_dir.iterdir()):
            if overwrite: shutil.rmtree(output_dir)
            else: raise FileExistsError(f"输出目录已存在且非空：{output_dir}。如需覆盖请加 --overwrite。")
        output_dir.mkdir(parents=True,exist_ok=True); self.records=[]; self.shard_index=0; self.num_scenes=0
    def add(self,record:Dict[str,Any])->None:
        self.records.append(record); self.num_scenes+=1
        if len(self.records)>=self.shard_size: self.flush()
    def flush(self)->None:
        if not self.records: return
        path=self.output_dir/f"shard_{self.shard_index:05d}.pt"
        torch.save({"format_version":2,"split":self.split,"prototype_bank":self.prototype_bank,"scenes":self.records},path)
        self.records=[]; self.shard_index+=1
    def close(self)->Dict[str,Any]:
        self.flush()
        return {"assignment_dir":str(self.output_dir.resolve()),"num_shards":self.shard_index,"num_scenes":self.num_scenes,"shard_size":self.shard_size}

@torch.inference_mode()
def process_split(split:str,model,datamodule,bank:Dict[str,Any],device:torch.device,args,output_dir:Path,residual_stats:ResidualStats|None)->Dict[str,Any]:
    prototype_latents=bank["prototype_latents_centered_raw"].float().to(device); prototype_groups=bank["prototype_coarse_ids"].long().to(device)
    num_coarse_groups=int(bank.get("num_coarse_groups",int(prototype_groups.max())+1)); use_global=num_coarse_groups==1
    prototype_traj=bank.get("prototype_trajectories_normalized")
    if args.compute_trajectory_metrics:
        if prototype_traj is None: raise KeyError("启用trajectory metrics时bank缺少 prototype_trajectories_normalized")
        prototype_traj=prototype_traj.float().to(device)
    coarse_centers=descriptor_stats=descriptor_cfg=None
    if not use_global:
        coarse_centers=bank["coarse_kmeans_centers_descriptor"].float().to(device); descriptor_stats=bank["descriptor_normalizer"]; descriptor_cfg=bank["descriptor_config"]
    margin_threshold=float(bank.get("support_margin_threshold",0.15)); temperature=bank.get("support_temperature_per_coarse",bank.get("support_temperature",1.0))
    metric={"count":0,"latent":0.0,"ade":0.0,"fde":0.0,"score":0.0,"boundary":0}; max_batches=args.max_train_batches if split=="train" else args.max_val_batches
    loader=get_loader(datamodule,split); writer=AssignmentShardWriter(output_dir,split,args.shard_size,str(Path(args.prototype_bank).expanduser().resolve()),args.overwrite); processed_scenarios=set()
    for batch_idx,data in enumerate(tqdm(loader,desc=f"Assign {split} -> sidecar shards")):
        if max_batches>0 and batch_idx>=max_batches: break
        data=data.to(device); target,predict_mask,valid,_=get_targets(model,data,args.agent_scope); n=target.size(0)
        primary=torch.full((n,),-1,dtype=torch.long,device=device); coarse_full=torch.full_like(primary,-1); second=torch.full_like(primary,-1)
        support_ids=torch.full((n,2),-1,dtype=torch.long,device=device); support_weights=torch.zeros((n,2),dtype=torch.float32,device=device); support_size=torch.zeros(n,dtype=torch.long,device=device)
        inf=torch.full((n,),float("inf"),device=device); d1=inf.clone(); d2=inf.clone(); margin=inf.clone(); boundary=torch.zeros(n,dtype=torch.bool,device=device)
        ade_full=inf.clone(); fde_full=inf.clone(); score_full=inf.clone(); z_full=torch.zeros((n,1,int(bank["latent_dim"])),dtype=torch.float32,device=device); residual_full=torch.zeros_like(z_full)
        if valid.any():
            target_valid=target[valid]; mask_valid=predict_mask[valid]
            with autocast_context(device,args.precision): z_valid=encode_centered_raw(model,target_valid,mask_valid).float()
            if use_global: coarse=torch.zeros(z_valid.size(0),dtype=torch.long,device=device); result=assign_global(z_valid,prototype_latents,args.predict_chunk,margin_threshold,temperature)
            else:
                blocks=build_trajectory_descriptor_blocks(target_valid,mask_valid,float(model.trajectory_scale),int(descriptor_cfg["dct_coeffs"]),int(descriptor_cfg["num_segments"]),float(descriptor_cfg["time_step"]),float(descriptor_cfg["slow_speed_threshold"]))
                descriptor=transform_descriptor_blocks(blocks,descriptor_stats); coarse=nearest_coarse_from_descriptor(descriptor,coarse_centers,args.predict_chunk)
                result=assign_hierarchical(z_valid,coarse,prototype_latents,prototype_groups,args.predict_chunk,margin_threshold,temperature)
            if args.compute_trajectory_metrics:
                selected=prototype_traj[result["primary"]]; ade,fde,score=trajectory_metrics(target_valid.float(),mask_valid,selected,float(model.trajectory_scale))
                ade_full[valid]=ade; fde_full[valid]=fde; score_full[valid]=score; metric["ade"]+=float(ade.sum()); metric["fde"]+=float(fde.sum()); metric["score"]+=float(score.sum())
            residual=z_valid[:,0]-prototype_latents[:,0][result["primary"]]
            primary[valid]=result["primary"]; coarse_full[valid]=coarse; second[valid]=result["second"]; support_ids[valid]=result["support_ids"]; support_weights[valid]=result["support_weights"]; support_size[valid]=result["support_size"]
            d1[valid]=result["distance1"]; d2[valid]=result["distance2"]; margin[valid]=result["margin"]; boundary[valid]=result["boundary"]; z_full[valid]=z_valid; residual_full[valid,0]=residual
            metric["count"]+=int(result["primary"].numel()); metric["latent"]+=float(result["distance1"].sum()); metric["boundary"]+=int(result["boundary"].sum())
            if residual_stats is not None: residual_stats.update(result["primary"],residual)
        ptr=get_agent_ptr(data); scenario_ids=get_scenario_ids(data,ptr.numel()-1)
        for scene_index in range(ptr.numel()-1):
            start,end=int(ptr[scene_index]),int(ptr[scene_index+1]); sid=scenario_ids[scene_index]
            if sid in processed_scenarios: raise RuntimeError(f"重复处理 scenario_id={sid}")
            fields={"coarse_index":coarse_full[start:end],"prototype_index":primary[start:end],"secondary_prototype_index":second[start:end],"support_prototype_ids":support_ids[start:end],"support_weights":support_weights[start:end],"support_size":support_size[start:end],"boundary_margin":margin[start:end],"is_boundary":boundary[start:end],"valid_agent_mask":valid[start:end],"match_latent_raw_l2":d1[start:end],"second_match_latent_raw_l2":d2[start:end],"match_ade_m":ade_full[start:end],"match_fde_m":fde_full[start:end],"match_traj_score_m":score_full[start:end],"z_gt_centered_raw":z_full[start:end],"z_residual":residual_full[start:end]}
            record={"scenario_id":sid,"num_agents":end-start}
            for key,value in fields.items(): record[key]=value.detach().cpu().contiguous()
            writer.add(record); processed_scenarios.add(sid)
    writer_info=writer.close(); count=max(metric["count"],1)
    return {"num_scenes_written":len(processed_scenarios),"num_valid_agents":metric["count"],"mean_match_latent_raw_l2":metric["latent"]/count,"mean_match_ade_m":metric["ade"]/count if args.compute_trajectory_metrics else None,"mean_match_fde_m":metric["fde"]/count if args.compute_trajectory_metrics else None,"mean_match_traj_score_m":metric["score"]/count if args.compute_trajectory_metrics else None,"boundary_agent_rate":metric["boundary"]/count,**writer_info}



def parse_args():
    parser=argparse.ArgumentParser(description="高速生成train/val prototype assignment sidecar shards；不复制或重写processed pkl。")
    parser.add_argument("--prototype_bank",required=True); parser.add_argument("--checkpoint",default=None,help="默认使用bank中记录的checkpoint。")
    parser.add_argument("--root",required=True); parser.add_argument("--out_root",required=True); parser.add_argument("--splits",nargs="+",choices=["train","val"],default=["train","val"])
    parser.add_argument("--agent_scope",choices=["all_valid","focal"],default=None,help="默认沿用bank配置。")
    parser.add_argument("--train_processed_dir",default=None); parser.add_argument("--val_processed_dir",default=None); parser.add_argument("--train_raw_dir",default=None); parser.add_argument("--val_raw_dir",default=None)
    parser.add_argument("--batch_size",type=int,default=64); parser.add_argument("--num_workers",type=int,default=8)
    parser.add_argument("--pin_memory",action=argparse.BooleanOptionalAction,default=True); parser.add_argument("--persistent_workers",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--device",default="auto"); parser.add_argument("--precision",choices=["32","bf16","fp16"],default="bf16"); parser.add_argument("--predict_chunk",type=int,default=131072)
    parser.add_argument("--shard_size",type=int,default=1024); parser.add_argument("--compute_trajectory_metrics",action=argparse.BooleanOptionalAction,default=True); parser.add_argument("--patch_dataset_fields",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--max_train_batches",type=int,default=0); parser.add_argument("--max_val_batches",type=int,default=0); parser.add_argument("--updated_bank_path",default=None)
    parser.add_argument("--seed",type=int,default=2030); parser.add_argument("--overwrite",action="store_true"); return parser.parse_args()



def main():
    args=parse_args(); set_seed(args.seed); patch_dataset_assignment_fields(args.patch_dataset_fields); device=resolve_device(args.device); bank_path=Path(args.prototype_bank).expanduser().resolve()
    bank=torch.load(bank_path,map_location="cpu",weights_only=False)
    required=["prototype_latents_centered_raw","prototype_coarse_ids","latent_dim","num_prototypes"]
    if args.compute_trajectory_metrics: required.append("prototype_trajectories_normalized")
    if int(bank.get("num_coarse_groups",1))>1: required+=["coarse_kmeans_centers_descriptor","descriptor_normalizer","descriptor_config"]
    for key in required:
        if key not in bank: raise KeyError(f"prototype bank缺少 {key}")
    checkpoint=args.checkpoint or bank.get("checkpoint")
    if not checkpoint: raise ValueError("未提供 --checkpoint，且bank中没有checkpoint。")
    model,hparams=load_vae_model(checkpoint,device)
    if int(bank["latent_dim"])!=int(model.latent_dim): raise RuntimeError(f"latent_dim不一致：bank={bank['latent_dim']}, model={model.latent_dim}")
    args.agent_scope=args.agent_scope or str(bank.get("agent_scope","all_valid")); datamodule=build_datamodule(hparams,args)
    out_root=Path(args.out_root).expanduser()
    if out_root.exists() and any(out_root.iterdir()) and args.overwrite: shutil.rmtree(out_root)
    out_root.mkdir(parents=True,exist_ok=True)
    manifest={"format_version":6,"mode":"prototype_assignment_sidecar_shards","prototype_bank":str(bank_path),"checkpoint":str(Path(checkpoint).expanduser().resolve()),"agent_scope":args.agent_scope,"num_coarse_groups":int(bank.get("num_coarse_groups",1)),"compute_trajectory_metrics":bool(args.compute_trajectory_metrics),"splits":{}}
    train_stats=ResidualStats(int(bank["num_prototypes"]),int(bank["latent_dim"])) if "train" in args.splits else None
    for split in args.splits:
        output=out_root/split
        stats=process_split(split,model,datamodule,bank,device,args,output,train_stats if split=="train" else None); manifest["splits"][split]=stats
        metric_text="" if stats["mean_match_ade_m"] is None else f", ADE={stats['mean_match_ade_m']:.6f}, FDE={stats['mean_match_fde_m']:.6f}"
        print(f"{split}: scenes={stats['num_scenes_written']:,}, shards={stats['num_shards']:,}, agents={stats['num_valid_agents']:,}, latent={stats['mean_match_latent_raw_l2']:.6f}{metric_text}, boundary={stats['boundary_agent_rate']:.2%}")
    if train_stats is not None:
        finalized=train_stats.finalize(); updated=dict(bank); updated.update(finalized); updated["assignment_statistics_attached"]=True; updated["assignment_statistics_source"]="prototype_assignment_sidecar_shards"
        updated_path=Path(args.updated_bank_path).expanduser() if args.updated_bank_path else bank_path.parent/"prototype_bank_with_train_stats.pt"
        torch.save(updated,updated_path); save_residual_stats_csv(out_root/"prototype_residual_stats.csv",finalized); manifest["updated_prototype_bank"]=str(updated_path.resolve()); print(f"带train residual统计的bank：{updated_path}")
    (out_root/"assignment_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); torch.save(manifest,out_root/"assignment_manifest.pt")
    print(f"\n完成。轻量assignment shards：{out_root}"); print("FM训练时将 prototype_assignment_dir 指向该目录；原processed pkl保持不变。Dataset字段补丁已在本脚本内处理。")



if __name__=="__main__":
    main()
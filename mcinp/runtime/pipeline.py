"""Stepwise formal entry point; data, each training job, and each evaluation split can be resumed and checked independently."""
import csv
from pathlib import Path
import numpy as np,torch
from .common import *
from .datasets import scenario_plan, iter_cases
from .equation import qualify,relative
from ..methods import METHODS, build as build_model
from ..methods import pod_linear
from ..methods.deeponet import fit_normalization as fit_don2_normalization

def data_binding(c):return dict(reference_source=identity({n:sha(Path(__file__).parent/n) for n in ('equation.py','datasets.py')}),generation_contract='paper-id-inputs-v1',equation='directional_signed_v2',dimension=c['dimension'],times=c['times'],reference_cap=c['reference_cap'],plan=identity(scenario_plan(c)),input_policy='model-f32-exact/truth-f64; noise-clean')
def model_config(c):
    keys=('dimension','N','B','L','width','modes','method','equation')
    result={k:c[k] for k in keys}
    if c['method']=='deeponet':
        result['don_width']=c.get('don_width',64 if c['dimension']==1 else 128)
        result['don_latent']=c.get('don_latent',64)
    return result
def training_config(c):
    keys=('seed','epochs','patience','batch_size','learning_rate','weight_decay','optimizer','shuffle')
    if c['method']=='deeponet' and c['dimension']==2:keys+=('minimum_epochs','min_relative_improvement')
    if c['method']=='pod_linear':keys+=('pod_ranks','pod_ridges')
    return {k:c[k] for k in keys}
def slot(c):return f"d{c['dimension']}_{c['method']}_s{c['seed']}_N{c['N']}_B{c['B']}_L{c['L']}_w{c['width']}_ep{c['epochs']}p{c['patience']}"
def training_binding(c,root):return dict(source=source_hash(),model=model_config(c),training=training_config(c),data=sha(root/'data'/'complete.json'),runtime=dict(torch=str(torch.__version__),device=c['device'],canonical_fft=True))
def data_check(c,root):
    verify(root/'data'/'complete.json',data_binding(c));manifest=load(root/'data'/'manifest.json')
    for row in manifest['splits']:
        d=root/'data'/row['split'];verify(d/'complete.json',dict(data=data_binding(c),scenario=row['scenario']))
        if sha(d/'complete.json')!=row['complete_sha256']:raise ValueError('split completion SHA')
        for r in load(d/'cases.json'):verify(d/r['receipt'])
    return manifest

def generate(a,c,root):
    out=root/'data';out.mkdir(parents=True,exist_ok=True);binding=data_binding(c);done=out/'complete.json'
    if done.exists():data_check(c,root);print('verified complete data; skip');return
    plan=scenario_plan(c);rows=[];start=timer();newcases=0;newsplit=0
    for row in plan:
        dst=out/row['split'];dst.mkdir(parents=True,exist_ok=True);b=dict(data=binding,scenario=row);complete=dst/'complete.json'
        if complete.exists():verify(complete,b);rows.append(dict(split=row['split'],scenario=row,complete_sha256=sha(complete)));continue
        cases=[]
        for i, case in enumerate(iter_cases(c['dimension'], row)):
            spec=case.pop('identity')
            times = np.array(c['times'])
            grid = 2 * np.pi * np.arange(row['N'], dtype=np.float64) / row['N']
            arrays = dict(**case, times=times, primary=np.ones(len(times), dtype=bool),
                          x=grid, y=grid.copy() if c['dimension'] == 2 else np.empty(0))
            inp={k:array_sha(v) for k,v in arrays.items()};cb=dict(data=binding,instance=spec,input_hashes=inp)
            p=dst/f'case_{i:04d}.npz';q=dst/f'case_{i:04d}.json'
            if q.exists():verify(q,cb);report=load(q)['qualification']
            else:
                # Noisy input variants share the same clean PDE-label cache.
                labelkeys=['truth_u0','truth_D','truth_alpha','alpha_after','switch','times'];labelkey=identity({k:inp[k] for k in labelkeys})
                labeldir=out/'labels';labeldir.mkdir(exist_ok=True);lp=labeldir/(labelkey+'.npz');lq=labeldir/(labelkey+'.json');lb=dict(data=binding,label=labelkey)
                if lq.exists():
                    lr=verify(lq,lb);truth=np.load(lp)['truth'];report=lr['qualification']
                else:
                    sw=float(case['switch']);st=timer()
                    truth,report=qualify(case['truth_u0'],case['truth_D'],case['truth_alpha'],times,'signed',c['reference_cap'],
                        switch=sw if sw>=0 else None,alphas_after=case['alpha_after'] if sw>=0 else None)
                    report['seconds']=elapsed(st);save_npz(lp,truth=truth)
                    receipt(lq,lb,[lp],qualification=report)
                save_npz(p,**arrays,truth=truth);receipt(q,cb,[p],qualification=report,instance=spec,truth_cache=labelkey)
                newcases+=1;print('qualified',row['split'],i,'N',row['N'],'error',report['relative_uncertainty_estimate'],flush=True)
            cases.append(dict(file=p.name,receipt=q.name,case_binding=cb))
            if a.stop_after_case and newcases>=a.stop_after_case:return
        write(dst/'cases.json',cases);receipt(complete,b,[dst/'cases.json']+[dst/r['receipt'] for r in cases],count=row['count'])
        rows.append(dict(split=row['split'],scenario=row,complete_sha256=sha(complete)));newsplit+=1
        if a.stop_after_split and newsplit>=a.stop_after_split:return
    write(out/'manifest.json',dict(splits=rows,input_policy=binding['input_policy']))
    receipt(done,binding,[out/'manifest.json'],seconds=elapsed(start),new_qualified_cases=newcases)

def cases_for(root,split):
    d=root/'data'/split
    for row in load(d/'cases.json'):
        verify(d/row['receipt'],row['case_binding']);yield np.load(d/row['file']),load(d/row['receipt'])
def role_split(manifest,role):return next(r['split'] for r in manifest['splits'] if r['scenario']['role']==role)
def queries(root,split):
    rows=[]
    for z,r in cases_for(root,split):
        for j,t in enumerate(z['times']):
            if t>0 and z['primary'][j]:rows.append((z['u0'],z['D'],z['alpha'],t,z['truth'][j]))
    return tuple(torch.tensor(np.asarray([r[i] for r in rows]),dtype=torch.float32) for i in range(5))
def train(a,c,root):
    manifest=data_check(c,root);b=training_binding(c,root);out=root/'slots'/slot(c);out.mkdir(parents=True,exist_ok=True);done=out/'complete.json'
    # N identifies the training grid; N128 is a native-grid test query in this contract and must not be labeled as a training grid.
    if c['N']!=next(r['scenario']['N'] for r in manifest['splits'] if r['scenario']['role']=='train'):raise ValueError('training N disagrees with data grid')
    if done.exists():verify(done,b);print('verified complete training; skip',slot(c));return
    start=timer(c['device']);seed_all(c['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trainq=queries(root,role_split(manifest,'train'));valq=queries(root,role_split(manifest,'validation'))
    if c['method']=='pod_linear':
        if c['dimension']==2 and c['N']!=64:raise ValueError('the 2-D POD baseline is fitted on the paper N64 training grid')
        # The closed-form fit lives with the method, because the basis, the normalization and the map are
        # all method definitions rather than training-loop mechanics.
        rec=pod_linear.fit(c['dimension'],c,trainq,valq,out/'pod.npz')
        write(out/'validation_selection.json',rec)
        receipt(done,b,[out/'pod.npz',out/'validation_selection.json'],seconds=elapsed(start,c['device']),parameters=0,
                rank=rec['rank'],ridge=rec['ridge'],stored_coefficients=rec.get('stored_coefficients'),normalization='train-only' if c['dimension']==2 else 'closed form')
        return
    norm=None
    if c['method']=='deeponet' and c['dimension']==2:
        normpath=out/'normalization.json';norm=fit_don2_normalization(root,normpath,role_split(manifest, 'train'))
    model=build_model(model_config(c),normalization=norm).to(c['device']);params=list(model.parameters())
    if not params:
        save_checkpoint(out/'selected.pt',dict(model=tensor_state(model),binding=b,config=model_config(c),epoch=0));receipt(done,b,[out/'selected.pt'],seconds=elapsed(start,c['device']),parameters=0);return
    opt=(torch.optim.AdamW if c['optimizer']=='AdamW' else torch.optim.Adam)(params,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    epoch=0;best=float('inf');bestepoch=0;history=[];changed=False;total_seconds=0.;resume=out/'resume.pt'
    is_don2=c['method']=='deeponet' and c['dimension']==2
    minimum_epochs=int(c.get('minimum_epochs',1)) if is_don2 else 1
    relative_threshold=float(c.get('min_relative_improvement',0.)) if is_don2 else 0.
    stagnant=0
    if resume.exists():
        z=torch.load(resume,map_location='cpu',weights_only=True)
        if z['binding']!=b:raise ValueError('resume identity')
        strict_load(model,z['model']);opt.load_state_dict(z['optimizer']);restore_rng(z['rng']);epoch=z['epoch'];best=z['best'];bestepoch=z['bestepoch'];history=z['history'];changed=z['changed'];total_seconds=z['seconds'];stagnant=z.get('stagnant',0)
        for state in opt.state.values():
            for k,v in state.items():
                # Keep the step tensor of non-capturable Adam/AdamW on the CPU after loading; move momentum tensors with the parameters.
                if isinstance(v,torch.Tensor) and k!='step':state[k]=v.to(c['device'])
    limit=min(c['epochs'],a.stop_after_epoch or c['epochs'])
    while epoch<limit and (epoch<minimum_epochs or stagnant<c['patience']):
        model.train();ids=torch.randperm(len(trainq[0])) if c['shuffle'] else torch.arange(len(trainq[0]));loss_sum=0.;gradnorm=0.;updates=0;before=[p.detach().clone() for p in params]
        for ii in ids.split(c['batch_size']):
            batch=[v[ii].to(c['device']) for v in trainq];opt.zero_grad();loss=(model(*batch[:4])-batch[4]).square().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite train loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('missing/nonfinite gradient')
            gradnorm+=sum(float(p.grad.abs().square().sum()) for p in params);opt.step();updates+=1;loss_sum+=float(loss.detach())*len(ii)
        changed=changed or any(not torch.equal(p,v) for p,v in zip(params,before));model.eval();valsum=0.
        with torch.no_grad():
            for ii in torch.arange(len(valq[0])).split(c['batch_size']):
                batch=[v[ii].to(c['device']) for v in valq];valsum+=float((model(*batch[:4])-batch[4]).square().mean())*len(ii)
        vl=valsum/len(valq[0]);epoch+=1
        if not np.isfinite(vl):raise ValueError('nonfinite validation')
        history.append(dict(epoch=epoch,train_mse=loss_sum/len(trainq[0]),validation_mse=vl,gradient_norm_squared=gradnorm,updates=updates,parameters_updated=changed))
        improved=vl<best*(1-relative_threshold) if np.isfinite(best) else True
        if improved:
            best=vl;bestepoch=epoch;stagnant=0;save_checkpoint(out/'selected.pt',dict(model=tensor_state(model),binding=b,config=model_config(c),epoch=epoch))
        else:stagnant+=1
        history[-1]['validation_improved']=improved;history[-1]['stagnant_epochs']=stagnant
        save_checkpoint(resume,dict(model=tensor_state(model),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,best=best,bestepoch=bestepoch,history=history,changed=changed,stagnant=stagnant,seconds=total_seconds+elapsed(start,c['device']),binding=b))
        print('epoch',slot(c),epoch,'validation',vl,flush=True)
    write_csv(out/'history.csv',history);write(out/'validation_selection.json',dict(bestepoch=bestepoch,validation_mse=best,test_used=False))
    if epoch==c['epochs'] or (epoch>=minimum_epochs and stagnant>=c['patience']):
        if not changed:raise ValueError('no trainable parameter updated')
        artifacts=[out/'selected.pt',resume,out/'history.csv',out/'validation_selection.json']
        if norm is not None:artifacts.append(out/'normalization.json')
        receipt(done,b,artifacts,seconds=total_seconds+elapsed(start,c['device']),parameters=sum(p.numel() for p in params),epochs=epoch)

def predictor(c,out):
    if c['method']=='pod_linear':return pod_linear.predictor(c['dimension'],c,out)
    norm=None
    if c['method']=='deeponet' and c['dimension']==2:norm=load(out/'normalization.json')
    model=build_model(model_config(c),normalization=norm).to(c['device']).eval();z=torch.load(out/'selected.pt',map_location='cpu',weights_only=True);strict_load(model,z['model'])
    def predict(u,d,alpha,t):
        with torch.no_grad():return model(*(torch.tensor(v,dtype=torch.float32,device=c['device']) for v in (u[None],d[None],alpha[None],np.array([t])))).cpu().numpy()[0]
    return predict
def predict_trajectory(predict,z):
    sw=float(z['switch']);state=None
    if sw>=0:state=predict(z['u0'],z['D'],z['alpha'],sw)
    return np.stack([predict(z['u0'],z['D'],z['alpha'],float(t)) if sw<0 or t<=sw else predict(state,z['D'],z['alpha_after'],float(t)-sw) for t in z['times']]),state
def metrics(pred,truth,times,primary,u0,uncertainty):
    axes=tuple(range(1,pred.ndim));targetmean=float(u0.mean());projected=pred.astype(float)-pred.mean(axes,keepdims=True,dtype=float)+targetmean
    pos=(times>0)&primary;terminal=np.where(primary)[0][-1];m={}
    for tag,v in [('raw',pred),('projected',projected)]:
        m[tag+'_trajectory']=relative(v[pos],truth[pos]);m[tag+'_terminal']=relative(v[terminal],truth[terminal]);m[tag+'_absolute_trajectory_rms']=float(np.sqrt(np.mean((v[pos]-truth[pos])**2)))
    m.update(raw_mass_error=float(max(abs(x.mean(dtype=float)-targetmean) for x in pred)),projected_mass_error=float(max(abs(x.mean()-targetmean) for x in projected)),t0_exact=bool(np.array_equal(pred[0],u0.astype(np.float32))),finite=bool(np.isfinite(pred).all()),reference_uncertainty=uncertainty)
    m['reference_to_raw_model_ratio']=uncertainty/max(m['raw_trajectory'],1e-15);m['reference_model_budget_resolved']=bool(m['reference_to_raw_model_ratio']<=.01)
    return m,projected
def evaluate(a,c,root):
    manifest=data_check(c,root);out=root/'slots'/slot(c);verify(out/'complete.json',training_binding(c,root));predict=predictor(c,out)
    checkpoint=out/('pod.npz' if c['method']=='pod_linear' else 'selected.pt');count=0
    for row in manifest['splits']:
        if row['scenario']['role']!='test' or (a.split and a.split!=row['split']):continue
        dest=out/'evaluation'/row['split'];dest.mkdir(parents=True,exist_ok=True);b=dict(training=training_binding(c,root),checkpoint=sha(checkpoint),split=sha(root/'data'/row['split']/'complete.json'),evaluation='piecewise-composed/noisy-input-clean-truth/native-grid-v2')
        done=dest/'complete.json'
        if done.exists():
            verify(done,b)
            for q in sorted(dest.glob('case_*.json')):verify(q)
            continue
        rows=[];start=timer(c['device'])
        for i,(z,qr) in enumerate(cases_for(root,row['split'])):
            p=dest/f'case_{i:04d}.npz';q=dest/f'case_{i:04d}.json';cb=dict(evaluation=b,case=qr['binding'])
            if q.exists():verify(q,cb);mm=load(q)['metrics']
            else:
                st=timer(c['device']);raw,state=predict_trajectory(predict,z);forward_seconds=elapsed(st,c['device']);mm,proj=metrics(raw,z['truth'],z['times'],z['primary'],z['u0'],qr['qualification']['relative_uncertainty_estimate'])
                if not mm['finite'] or not mm['t0_exact']:raise ValueError('prediction finite/t0 gate')
                mm['forward_seconds']=forward_seconds;save_npz(p,raw=raw,projected=proj,truth=z['truth'],times=z['times'],primary=z['primary'],x=z['x'],y=z['y'],u0=z['u0'],truth_u0=z['truth_u0'],predicted_switch_state=state if state is not None else np.empty(0))
                receipt(q,cb,[p],metrics=mm)
            rows.append(dict(slot=slot(c),epochs=c['epochs'],split=row['split'],index=i,N=row['scenario']['N'],method=METHODS[c['method']][0],seed=c['seed'],B=c['B'],L=c['L'],width=c['width'],dimension=c['dimension'],**mm))
        write(dest/'metrics.json',rows);receipt(done,b,[dest/'metrics.json']+[p for p in sorted(dest.glob('case_*.json'))],seconds=elapsed(start,c['device']));count+=1
        if a.stop_after_split and count>=a.stop_after_split:return

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def export_metrics(c, root):
    out = root / 'slots' / slot(c)
    verify(out / 'complete.json', training_binding(c, root))
    rows = []
    for completion in sorted((out / 'evaluation').glob('*/complete.json')):
        verify(completion)
        rows.extend(load(completion.parent / 'metrics.json'))
    if not rows:
        raise ValueError('no completed evaluation splits found')
    path = out / 'per_instance_metrics.csv'
    write_csv(path, rows)
    return path
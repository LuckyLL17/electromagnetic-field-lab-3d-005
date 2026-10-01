import { useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

type Mode='dipole'|'coil'|'capacitor';
interface DipoleParams{strength:number;separation:number;range:number}
interface CoilParams{strength:number;turns:number;radius:number}
interface CapacitorParams{strength:number;gap:number}
type AnyParams=DipoleParams|CoilParams|CapacitorParams;
interface ParamsByMode{dipole:DipoleParams;coil:CoilParams;capacitor:CapacitorParams}

const DEFAULTS:ParamsByMode={
  dipole:{strength:1.6,separation:1.6,range:2.6},
  coil:{strength:1.6,turns:7,radius:1.55},
  capacitor:{strength:1.6,gap:6},
};
const LINE_COUNT:Record<Mode,number>={dipole:18,coil:12,capacitor:9};
const MODE_LABEL:Record<Mode,string>={dipole:'电偶极子',coil:'螺线管',capacitor:'平行板电容器'};
const colors={field:'#68d7f2',vector:'#f6c86f',positive:'#f27f73',negative:'#6a9fff',coil:'#d995f2'};
const COIL_SPACING=.34,PLATE_HALF=2.6,CHARGE_R=.28;

/* ---------- 场函数（参数化） ---------- */
// 完全椭圆积分 K(m)、E(m)，参数 m = k^2；K 用 AGM，E 由 E=(1-m)(K+2m·dK/dm) 中心差分求得
function kagm(m:number){
  const mm=Math.min(Math.max(m,0),1-1e-12);
  let a=1,g=Math.sqrt(1-mm);
  for(let i=0;i<16&&a-g>1e-16;i++){const an=(a+g)/2;g=Math.sqrt(a*g);a=an;}
  return Math.PI/(2*a);
}
function ellipticKE(m:number):[number,number]{
  const mm=Math.min(Math.max(m,0),1-1e-12);
  const h=1e-6;
  const m1=Math.min(Math.max(mm-h,0),1-1e-12),m2=Math.min(Math.max(mm+h,0),1-1e-12);
  const K=kagm(mm);
  const dK=(kagm(m2)-kagm(m1))/(m2-m1);
  return [K,(1-mm)*(K+2*mm*dK)];
}
// 半径 R、位于 z0 的单个载流圆环在柱坐标 (rho,z) 处的磁场（μ0·I 归一）
function loopField(rho:number,z:number,R:number,z0:number):[number,number]{
  const dz=z-z0;
  if(rho<1e-6){const s=R*R+dz*dz;return [0,R*R/(2*Math.pow(s,1.5))];}
  const a2=(R+rho)*(R+rho)+dz*dz;
  const [K,E]=ellipticKE(4*R*rho/a2);
  const d=(R-rho)*(R-rho)+dz*dz;
  const sq=Math.sqrt(a2),f=1/(2*Math.PI);
  return [f*dz/(rho*sq)*(-K+E*(R*R+rho*rho+dz*dz)/d), f/sq*(K+E*(R*R-rho*rho-dz*dz)/d)];
}
function dipoleField(x:number,y:number,z:number,p:DipoleParams){
  const hx=p.separation/2,soft=.02;
  const ax=x+hx,bx=x-hx;
  const ra=Math.pow(ax*ax+y*y+z*z+soft,1.5),rb=Math.pow(bx*bx+y*y+z*z+soft,1.5);
  const k=p.strength;
  return new THREE.Vector3(k*(ax/ra-bx/rb),k*y*(1/ra-1/rb),k*z*(1/ra-1/rb));
}
function coilField(x:number,y:number,z:number,p:CoilParams){
  const rho=Math.hypot(x,y);
  let br=0,bz=0;
  for(let i=0;i<p.turns;i++){
    const [lr,lz]=loopField(rho,z,p.radius,(i-(p.turns-1)/2)*COIL_SPACING);
    br+=lr;bz+=lz;
  }
  const k=p.strength*2.2,s=rho<1e-6?0:1/rho;
  return new THREE.Vector3(x*s*br*k,y*s*br*k,bz*k);
}
function capacitorField(x:number,y:number,p:CapacitorParams){
  const hx=p.gap/2;
  const sx=1/(1+Math.exp((Math.abs(x)-hx)/.35));
  const dy=Math.abs(y)-PLATE_HALF;
  const fringe=Math.exp(-dy*dy/.6)*sx;
  return new THREE.Vector3(p.strength*(.12+.88*sx),p.strength*.2*fringe*Math.sign(y),0);
}
function fieldAt(x:number,y:number,z:number,mode:Mode,params:AnyParams){
  if(mode==='coil')return coilField(x,y,z,params as CoilParams);
  if(mode==='capacitor')return capacitorField(x,y,params as CapacitorParams);
  return dipoleField(x,y,z,params as DipoleParams);
}
// 有效采样范围：观察范围至少包含电荷本身
const dipoleExtent=(p:DipoleParams)=>Math.max(p.range,p.separation/2+.7);
const coilHalfLen=(p:CoilParams)=>(p.turns-1)*COIL_SPACING/2;

/* ---------- 场线追踪（RK2 中点法，固定步长，越界或到达负电荷即截断） ---------- */
function trace(seed:THREE.Vector3,mode:Mode,params:AnyParams,step:number,maxSteps:number,bounds:number,stop?:(q:THREE.Vector3)=>boolean){
  const pts:THREE.Vector3[]=[];
  const p=seed.clone();
  for(let i=0;i<maxSteps;i++){
    pts.push(p.clone());
    if(stop&&stop(p))break;
    const v1=fieldAt(p.x,p.y,p.z,mode,params),l1=v1.length();
    if(l1<1e-7)break;
    const mx=p.x+v1.x/l1*step/2,my=p.y+v1.y/l1*step/2,mz=p.z+v1.z/l1*step/2;
    const v2=fieldAt(mx,my,mz,mode,params),l2=v2.length();
    const d=l2>1e-7?v2:v1,dl=l2>1e-7?l2:l1;
    p.x+=d.x/dl*step;p.y+=d.y/dl*step;p.z+=d.z/dl*step;
    if(Math.max(Math.abs(p.x),Math.abs(p.y),Math.abs(p.z))>bounds)break;
  }
  return pts;
}
function buildLines(mode:Mode,params:AnyParams){
  const lines:THREE.Vector3[][]=[];
  if(mode==='dipole'){
    const p=params as DipoleParams,hx=p.separation/2,bounds=dipoleExtent(p);
    const stopR=Math.min(CHARGE_R+.05,hx-.02);
    const stop=(q:THREE.Vector3)=>{const dx=q.x-hx;return dx*dx+q.y*q.y+q.z*q.z<stopR*stopR};
    for(let k=0;k<6;k++){
      const th=(25+k*25)*Math.PI/180,r0=CHARGE_R+.06;
      const base=trace(new THREE.Vector3(-hx+Math.cos(th)*r0,Math.sin(th)*r0,0),mode,params,.05,Math.ceil(bounds*90),bounds,stop);
      for(const phi of [0,Math.PI*2/3,Math.PI*4/3]){
        if(phi===0){lines.push(base);continue;}
        const c=Math.cos(phi),s=Math.sin(phi);
        lines.push(base.map(q=>new THREE.Vector3(q.x,q.y*c-q.z*s,q.y*s+q.z*c)));
      }
    }
    return lines;
  }
  if(mode==='coil'){
    const p=params as CoilParams,hl=coilHalfLen(p);
    const bounds=3.2*Math.max(p.radius,hl)+.8,maxSteps=Math.ceil(6*bounds/.07);
    // 近轴线从下端穿入后延伸至远处；其余种子在中截面闭合环绕
    const seeds:[number,number][]=[[.3,-(hl+.5)],[.6,0],[.95,0],[1.15,0]];
    for(const [f,z0] of seeds){
      const base=trace(new THREE.Vector3(p.radius*f,0,z0),mode,params,.07,maxSteps,bounds);
      for(const phi of [0,Math.PI*2/3,Math.PI*4/3]){
        if(phi===0){lines.push(base);continue;}
        const c=Math.cos(phi),s=Math.sin(phi);
        lines.push(base.map(q=>new THREE.Vector3(q.x*c-q.y*s,q.x*s+q.y*c,q.z)));
      }
    }
    return lines;
  }
  const p=params as CapacitorParams,hx=p.gap/2;
  for(let k=0;k<9;k++){
    const y=-2.2+k*.55;
    lines.push(trace(new THREE.Vector3(-hx+.12,y,0),mode,params,.06,Math.ceil(p.gap/.06)+40,Math.max(hx,PLATE_HALF)+1.2,q=>q.x>=hx-.14));
  }
  return lines;
}
function buildVectors(mode:Mode,params:AnyParams){
  const n=5;
  let exA:[number,number],exB:[number,number],plane:'xy'|'xz'='xy';
  if(mode==='coil'){const p=params as CoilParams;const hl=coilHalfLen(p);exA=[-(p.radius+1.1),p.radius+1.1];exB=[-(hl+1.1),hl+1.1];plane='xz';}
  else if(mode==='capacitor'){const p=params as CapacitorParams;exA=[-(p.gap/2+.9),p.gap/2+.9];exB=[-2.9,2.9];}
  else{const e=dipoleExtent(params as DipoleParams);exA=[-e,e];exB=[-e,e];}
  const vecs:[THREE.Vector3,THREE.Vector3][]=[];
  const sa=(exA[1]-exA[0])/(n-1),sb=(exB[1]-exB[0])/(n-1);
  const len=Math.min(.42,.42*Math.min(sa,sb));
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const a=exA[0]+i*sa,b=exB[0]+j*sb;
    const x=a,y=plane==='xy'?b:0,z=plane==='xy'?0:b;
    const v=fieldAt(x,y,z,mode,params);
    if(v.length()<1e-7)continue;
    const start=new THREE.Vector3(x,y,z);
    vecs.push([start,start.clone().add(v.multiplyScalar(len/v.length()))]);
  }
  return vecs;
}

/* ---------- 场景组件 ---------- */
function Source({mode,params}:{mode:Mode;params:AnyParams}){
  if(mode==='capacitor'){
    const p=params as CapacitorParams,hx=p.gap/2;
    return <group>
      <mesh position={[-hx,0,0]}><boxGeometry args={[.16,5.2,.55]}/><meshStandardMaterial color={colors.positive}/></mesh>
      <mesh position={[hx,0,0]}><boxGeometry args={[.16,5.2,.55]}/><meshStandardMaterial color={colors.negative}/></mesh>
      <Text position={[-hx+.18,2.75,.2]} fontSize={.2} color={colors.positive}>+Q</Text>
      <Text position={[hx-.18,2.75,.2]} fontSize={.2} color={colors.negative}>−Q</Text>
    </group>;
  }
  if(mode==='coil'){
    const p=params as CoilParams;
    return <group>{Array.from({length:p.turns},(_,i)=><mesh key={i} position={[0,0,(i-(p.turns-1)/2)*COIL_SPACING]} rotation={[Math.PI/2,0,0]}><torusGeometry args={[p.radius,.07,10,48]}/><meshStandardMaterial color={colors.coil} emissive={colors.coil} emissiveIntensity={p.strength*.12}/></mesh>)}</group>;
  }
  const p=params as DipoleParams,hx=p.separation/2;
  return <group>
    <mesh position={[-hx,0,0]}><sphereGeometry args={[CHARGE_R,20,16]}/><meshStandardMaterial color={colors.positive} emissive={colors.positive} emissiveIntensity={.4}/></mesh>
    <mesh position={[hx,0,0]}><sphereGeometry args={[CHARGE_R,20,16]}/><meshStandardMaterial color={colors.negative} emissive={colors.negative} emissiveIntensity={.4}/></mesh>
    <Line points={[new THREE.Vector3(-hx,0,0),new THREE.Vector3(hx,0,0)]} color="#d8e4ef" lineWidth={2}/>
  </group>;
}
function FieldLines({mode,params}:{mode:Mode;params:AnyParams}){
  const lines=useMemo(()=>buildLines(mode,params),[mode,params]);
  return <group>{lines.map((pts,i)=><Line key={i} points={pts} color={mode==='coil'?colors.coil:colors.field} lineWidth={1.6} transparent opacity={.78}/>)}</group>;
}
function Vectors({mode,params,showVectors}:{mode:Mode;params:AnyParams;showVectors:boolean}){
  const vectors=useMemo(()=>buildVectors(mode,params),[mode,params]);
  return <group>{showVectors&&vectors.map((v,i)=><Line key={i} points={v} color={colors.vector} lineWidth={1.3}/>)}</group>;
}
function EMScene({mode,params,showVectors}:{mode:Mode;params:AnyParams;showVectors:boolean}){
  return <Canvas camera={{position:[6,4.5,8],fov:42}}>
    <color attach="background" args={['#0b0b1c']}/>
    <ambientLight intensity={1.2}/>
    <pointLight position={[3,4,5]} color="#9a8cff" intensity={18}/>
    <pointLight position={[-4,-2,2]} color="#49d5e7" intensity={12}/>
    <group key={mode} rotation={[.18,-.24,0]}>
      <Source mode={mode} params={params}/>
      <FieldLines mode={mode} params={params}/>
      <Vectors mode={mode} params={params} showVectors={showVectors}/>
      <mesh position={[0,0,-.45]}><planeGeometry args={[7,6]}/><meshBasicMaterial color="#25234b" transparent opacity={.12} side={THREE.DoubleSide}/></mesh>
    </group>
    <OrbitControls makeDefault minDistance={5} maxDistance={14}/>
  </Canvas>;
}

/* ---------- UI 组件 ---------- */
function Metric({label,value,note}:{label:string;value:string;note:string}){return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>}
function Slider({label,display,value,min,max,step,onChange}:{label:string;display:string;value:number;min:number;max:number;step:number;onChange:(v:number)=>void}){
  return <label className="control"><span>{label} <b>{display}</b></span><input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/></label>;
}
export default function App(){
  const [mode,setMode]=useState<Mode>('dipole');
  const [paramsMap,setParamsMap]=useState<ParamsByMode>(DEFAULTS);
  const [showVectors,setShowVectors]=useState(true);
  const params=paramsMap[mode];
  const modeLabel=MODE_LABEL[mode];
  const update=(patch:Partial<DipoleParams>&Partial<CoilParams>&Partial<CapacitorParams>)=>setParamsMap(prev=>{
    if(mode==='dipole')return {...prev,dipole:{...prev.dipole,...patch}};
    if(mode==='coil')return {...prev,coil:{...prev.coil,...patch}};
    return {...prev,capacitor:{...prev.capacitor,...patch}};
  });
  const reset=()=>setParamsMap(prev=>{
    if(mode==='dipole')return {...prev,dipole:{...DEFAULTS.dipole}};
    if(mode==='coil')return {...prev,coil:{...DEFAULTS.coil}};
    return {...prev,capacitor:{...DEFAULTS.capacitor}};
  });
  const dp=paramsMap.dipole,cp=paramsMap.coil,pp=paramsMap.capacitor;
  const metrics:[string,string,string][]=
    mode==='dipole'?[['场强系数',dp.strength.toFixed(1),'归一化激励'],['电荷间距',dp.separation.toFixed(1),'d · 决定偶极矩'],['观察范围','±'+dp.range.toFixed(1),'R · 场线截断半径']]
    :mode==='coil'?[['场强系数',cp.strength.toFixed(1),'归一化激励'],['线圈层数',String(cp.turns),'N · 匝数'],['线圈半径',cp.radius.toFixed(2),'r · 决定束流截面']]
    :[['场强系数',pp.strength.toFixed(1),'归一化激励'],['板间距',pp.gap.toFixed(1),'d · 匀强区宽度'],['场线束',String(LINE_COUNT.capacitor),'采样轨迹']];
  const readout=
    mode==='dipole'?{type:'静电场 · 非匀强',dir:'+ → −',geom:`间距 d = ${dp.separation.toFixed(1)} · 范围 R = ${dp.range.toFixed(1)}`,grid:`5 × 5 · ±${dipoleExtent(dp).toFixed(1)}（z = 0 截面）`,lines:`${LINE_COUNT.dipole} 束 · 截断于 ±${dipoleExtent(dp).toFixed(1)}`}
    :mode==='coil'?{type:'静磁场 · 近似',dir:'右手螺旋定则',geom:`层数 N = ${cp.turns} · 半径 r = ${cp.radius.toFixed(2)}`,grid:`5 × 5 · ±${(cp.radius+1.1).toFixed(1)} × ±${(coilHalfLen(cp)+1.1).toFixed(1)}（轴向截面）`,lines:`${LINE_COUNT.coil} 束 · 闭合磁感线`}
    :{type:'静电场 · 匀强',dir:'+ → −',geom:`板间距 d = ${pp.gap.toFixed(1)}`,grid:`5 × 5 · ±${(pp.gap/2+.9).toFixed(1)} × ±2.9（z = 0 截面）`,lines:`${LINE_COUNT.capacitor} 束 · 板间直线`};
  return <div className="app">
    <header className="topbar">
      <div className="brand"><div className="brand-icon">✦</div><div><span className="eyebrow">ELECTROMAGNETISM / FIELD EXPLORER</span><h1>ElectroMag Lab <span>·</span> 电磁场与磁力线</h1></div></div>
      <div className="header-meta"><span className="dataset">MODEL 03 / {modeLabel}</span><button onClick={()=>setShowVectors(v=>!v)}>{showVectors?'隐藏场矢量':'显示场矢量'}</button></div>
    </header>
    <main className="layout">
      <section className="model-card">
        <div className="model-head">
          <div><span className="eyebrow">3D FIELD MAPPING</span><h2>{modeLabel} · Field Line Tracer</h2></div>
          <div className="mode-switch">{(['dipole','coil','capacitor'] as Mode[]).map(item=><button key={item} className={mode===item?'active':''} onClick={()=>setMode(item)}>{item==='dipole'?'电偶极子':item==='coil'?'螺线管':'平行板'}</button>)}</div>
        </div>
        <div className="canvas-wrap">
          <EMScene mode={mode} params={params} showVectors={showVectors}/>
          <div className="viewer-legend"><span><i className="dot protein"/>场线</span><span><i className="dot ligand"/>场源</span><span><i className="dot interaction"/>矢量采样</span></div>
          <div className="viewer-hint">左键旋转 · 滚轮缩放 · 切换场源模型</div>
        </div>
      </section>
      <aside className="sidebar">
        <section className="panel intro-panel">
          <span className="eyebrow">FIELD EXPERIMENT</span>
          <h3>三维场线实验台</h3>
          <p>在电偶极子、螺线管与平行板电容器之间切换，调整各模型专属的几何参数（按模型独立保存），观察电场/磁场的空间拓扑和方向分布。</p>
          <div className="metrics">{metrics.map(m=><Metric key={m[0]} label={m[0]} value={m[1]} note={m[2]}/>)}</div>
        </section>
        <section className="panel">
          <div className="panel-title">
            <div><span className="eyebrow">SOURCE CONTROL</span><h3>场源与参数</h3></div>
            <button className="reset-btn" onClick={reset} title="仅重置当前模型的参数，其他模型不受影响">重置当前模型</button>
          </div>
          <Slider label="激励强度" display={`${params.strength.toFixed(1)} ×`} value={params.strength} min={.5} max={3} step={.1} onChange={v=>update({strength:v})}/>
          {mode==='dipole'&&<>
            <Slider label="电荷间距" display={dp.separation.toFixed(1)} value={dp.separation} min={.8} max={3} step={.1} onChange={v=>update({separation:v})}/>
            <Slider label="观察范围" display={`± ${dp.range.toFixed(1)}`} value={dp.range} min={1.5} max={4} step={.1} onChange={v=>update({range:v})}/>
          </>}
          {mode==='coil'&&<>
            <Slider label="线圈层数" display={`${cp.turns} 匝`} value={cp.turns} min={3} max={12} step={1} onChange={v=>update({turns:Math.round(v)})}/>
            <Slider label="线圈半径" display={cp.radius.toFixed(2)} value={cp.radius} min={.9} max={2.2} step={.05} onChange={v=>update({radius:v})}/>
          </>}
          {mode==='capacitor'&&<>
            <Slider label="板间距" display={pp.gap.toFixed(1)} value={pp.gap} min={2} max={8} step={.2} onChange={v=>update({gap:v})}/>
          </>}
          <div className="source-list">{(['dipole','coil','capacitor'] as Mode[]).map(item=><button key={item} className={mode===item?'source active':'source'} onClick={()=>setMode(item)}><span className={`source-dot ${item}`}/>{item==='dipole'?'电偶极子':item==='coil'?'通电螺线管':'平行板电容器'}<em>{item==='coil'?'B':'E'}</em></button>)}</div>
        </section>
        <section className="panel">
          <span className="eyebrow">FIELD READOUT</span>
          <div className="readout">
            <div><span>当前场型</span><b>{readout.type}</b></div>
            <div><span>方向约定</span><b>{readout.dir}</b></div>
            <div><span>几何参数</span><b>{readout.geom}</b></div>
            <div><span>采样网格</span><b>{showVectors?readout.grid:`已隐藏 · ${readout.grid}`}</b></div>
            <div><span>场线束</span><b>{readout.lines}</b></div>
            <div><span>模型精度</span><b>教学级采样</b></div>
          </div>
          <p className="readout-note">同一采样网格逐点对当前场函数求值：几何参数改变场分布与有效采样范围，因此同一网格会给出不同的矢量与场线结果。</p>
        </section>
      </aside>
    </main>
  </div>;
}

import { useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

type Mode = 'dipole' | 'coil' | 'capacitor';
const MODES: Mode[] = ['dipole', 'coil', 'capacitor'];
const MODE_NAME: Record<Mode, string> = { dipole: '电偶极子', coil: '螺线管', capacitor: '平行板电容器' };
const MODE_SHORT: Record<Mode, string> = { dipole: '电偶极子', coil: '螺线管', capacitor: '平行板' };
const FIELD_KIND: Record<Mode, string> = { dipole: '静电场 · 偶极型', coil: '静磁场 · 螺线管', capacitor: '静电场 · 匀强型' };
const LINE_COUNT: Record<Mode, number> = { dipole: 16, coil: 6, capacitor: 8 };

interface DipoleCfg { strength: number; separation: number; range: number }
interface CoilCfg { strength: number; layers: number; radius: number }
interface CapCfg { strength: number; gap: number }
type Cfg = DipoleCfg | CoilCfg | CapCfg;
interface Store { dipole: DipoleCfg; coil: CoilCfg; capacitor: CapCfg }

const DEFAULTS: Store = {
  dipole: { strength: 1.6, separation: 1.6, range: 4.2 },
  coil: { strength: 1.6, layers: 7, radius: 1.55 },
  capacitor: { strength: 1.6, gap: 6 },
};

const colors = { field: '#68d7f2', vector: '#f6c86f', positive: '#f27f73', negative: '#6a9fff', coil: '#d995f2' };

/* ---------- 场函数：形态参数直接进入采样公式 ---------- */

// 电偶极子：+q 位于 x=-d/2（左，红），-q 位于 x=+d/2（右，蓝），软化为教学级有限值
function dipoleE(x: number, y: number, z: number, c: DipoleCfg) {
  const a = c.separation / 2, s = c.strength * 1.6, eps2 = 0.05;
  let px = x + a, py = y, pz = z;
  let d3 = Math.pow(px * px + py * py + pz * pz + eps2, 1.5);
  let ex = s * px / d3, ey = s * py / d3, ez = s * pz / d3;
  px = x - a; py = y; pz = z;
  d3 = Math.pow(px * px + py * py + pz * pz + eps2, 1.5);
  ex -= s * px / d3; ey -= s * py / d3; ez -= s * pz / d3;
  return { x: ex, y: ey, z: ez };
}

// 螺线管：轴线沿 z，线圈半径 R、层数决定半高 h；双极子模型 + 内部均匀轴向场
function coilB(x: number, y: number, z: number, c: CoilCfg) {
  const R = c.radius, half = c.layers * 0.34 / 2, s = c.strength * 1.1;
  let bx = 0, by = 0, bz = 0;
  for (const sign of [1, -1]) {
    const dz = z - sign * half;
    const d3 = Math.pow(x * x + y * y + dz * dz + 0.06, 1.5);
    bx += s * x / d3; by += s * y / d3; bz += s * dz / d3;
  }
  const rho = Math.sqrt(x * x + y * y);
  if (rho < R * 1.02 && Math.abs(z) < half + 0.15) {
    bz += s * 0.42 * Math.max(0, 1 - 0.5 * (rho / R) ** 2);
  }
  return { x: bx, y: by, z: bz };
}

// 平行板电容器：+Q 板位于 x=-D/2，-Q 板位于 x=+D/2；板间近匀强（沿 +x），板外为快速衰减的边缘场
function capE(x: number, y: number, z: number, c: CapCfg) {
  const h = c.gap / 2, s = c.strength;
  if (Math.abs(x) <= h) {
    return { x: s * (0.9 + 0.08 * Math.cos(y * 1.2)), y: 0, z: 0 };
  }
  const sign = x > 0 ? 1 : -1;
  const frame = Math.exp(-(y * y + z * z) / 8.4);
  const fade = Math.exp(-(Math.abs(x) - h) / 1.4);
  // 板外边缘场：外侧被相邻极板吸回（沿 -x），y 分量向轴线弯折
  return { x: -s * 0.45 * fade * frame, y: -s * 0.16 * sign * y * fade * frame, z: 0 };
}

function fieldAt(x: number, y: number, z: number, mode: Mode, cfg: Cfg): THREE.Vector3 {
  if (mode === 'capacitor') {
    const f = capE(x, y, z, cfg as CapCfg);
    return new THREE.Vector3(f.x, f.y, f.z);
  }
  if (mode === 'coil') {
    const f = coilB(x, y, z, cfg as CoilCfg);
    return new THREE.Vector3(f.x, f.y, f.z);
  }
  const f = dipoleE(x, y, z, cfg as DipoleCfg);
  return new THREE.Vector3(f.x, f.y, f.z);
}

// 采样域（场线与矢量的有效范围），随形态参数更新
function sampleDomain(mode: Mode, cfg: Cfg): number {
  if (mode === 'coil') return (cfg as CoilCfg).radius + 0.9;
  if (mode === 'capacitor') return (cfg as CapCfg).gap / 2 + 0.9;
  return (cfg as DipoleCfg).range;
}

/* ---------- 场线 ---------- */

// RK4 沿单位方向场追踪偶极子场线
function traceDipole(c: DipoleCfg, seed: [number, number, number], dir: 1 | -1): THREE.Vector3[] {
  const a = c.separation / 2, h = 0.05 * dir, lim = c.range * 1.15;
  const p = [seed[0], seed[1], seed[2]];
  const pts: THREE.Vector3[] = [];
  const deriv = (q: number[]) => {
    const f = dipoleE(q[0], q[1], q[2], c);
    const m = Math.hypot(f.x, f.y, f.z) || 1e-6;
    return [f.x / m, f.y / m, f.z / m];
  };
  for (let i = 0; i < 430; i++) {
    pts.push(new THREE.Vector3(p[0], p[1], p[2]));
    const k1 = deriv(p);
    const k2 = deriv([p[0] + h / 2 * k1[0], p[1] + h / 2 * k1[1], p[2] + h / 2 * k1[2]]);
    const k3 = deriv([p[0] + h / 2 * k2[0], p[1] + h / 2 * k2[1], p[2] + h / 2 * k2[2]]);
    const k4 = deriv([p[0] + h * k3[0], p[1] + h * k3[1], p[2] + h * k3[2]]);
    for (let j = 0; j < 3; j++) p[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    const dTarget = Math.hypot(p[0] - dir * a, p[1], p[2]);
    if (dTarget < 0.24 || Math.hypot(p[0], p[1], p[2]) > lim) break;
  }
  return pts;
}

function dipoleLines(c: DipoleCfg): THREE.Vector3[][] {
  const out: THREE.Vector3[][] = [];
  const a = c.separation / 2, sr = a + 0.12, N = LINE_COUNT.dipole;
  for (let i = 0; i < N; i++) { // Fibonacci 球面种子，绕正电荷出发
    const phi = Math.acos(1 - 2 * (i + 0.5) / N);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const sp = Math.sin(phi);
    const seed: [number, number, number] = [-a + sr * sp * Math.cos(theta), sr * Math.cos(phi), sr * sp * Math.sin(theta)];
    const line = traceDipole(c, seed, 1);
    if (line.length > 3) out.push(line);
  }
  return out;
}

// 螺线管子午面闭合回路（内部沿轴 → 端部折返 → 外部闭合），再绕轴旋转成 3D 环
function coilLines(c: CoilCfg): THREE.Vector3[][] {
  const R = c.radius, half = c.layers * 0.34 / 2;
  const out: THREE.Vector3[][] = [];
  const N = LINE_COUNT.coil, M = 8;
  for (let k = 0; k < N; k++) {
    const rIn = R * (k + 1) / (N + 1);
    const rOut = R + 0.55 + k * 0.12;
    const cx = (rIn + rOut) / 2, rx = (rOut - rIn) / 2, cap = 0.32;
    const profile: [number, number][] = [];
    for (let i = 0; i <= 14; i++) profile.push([rIn, half - 2 * half * i / 14]);
    for (let i = 1; i <= 10; i++) { const t = Math.PI * (1 - i / 10); profile.push([cx + rx * Math.cos(t), -half - cap * Math.sin(t)]); }
    for (let i = 1; i <= 14; i++) profile.push([rOut, -half + 2 * half * i / 14]);
    for (let i = 1; i <= 10; i++) { const t = Math.PI * i / 10; profile.push([cx + rx * Math.cos(t), half + cap * Math.sin(t)]); }
    for (let j = 0; j < M; j++) {
      const ang = j / M * Math.PI * 2;
      out.push(profile.map(([r, z]) => new THREE.Vector3(r * Math.cos(ang), r * Math.sin(ang), z)));
    }
  }
  return out;
}

function capacitorLines(c: CapCfg): THREE.Vector3[][] {
  const h = c.gap / 2;
  // 匀强场区：场线沿 +x 从 +Q 板连向 -Q 板，间距随板间距自适应
  return Array.from({ length: LINE_COUNT.capacitor }, (_, i) => {
    const y = -1.75 + i * 3.5 / (LINE_COUNT.capacitor - 1);
    return [new THREE.Vector3(-h + .22, y, 0), new THREE.Vector3(h - .22, y, 0)];
  });
}

function FieldLines({ mode, cfg }: { mode: Mode; cfg: Cfg }) {
  const lines = useMemo(() => {
    if (mode === 'coil') return coilLines(cfg as CoilCfg);
    if (mode === 'capacitor') return capacitorLines(cfg as CapCfg);
    return dipoleLines(cfg as DipoleCfg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode + JSON.stringify(cfg)]);
  return (
    <group>
      {lines.map((line, i) => (
        <Line key={i} points={line} color={mode === 'coil' ? colors.coil : colors.field} lineWidth={1.6} transparent opacity={.78} />
      ))}
    </group>
  );
}

/* ---------- 矢量采样网格：有效范围随参数更新 ---------- */

function Vectors({ mode, cfg, showVectors }: { mode: Mode; cfg: Cfg; showVectors: boolean }) {
  const vectors = useMemo(() => {
    const d = sampleDomain(mode, cfg);
    const out: [THREE.Vector3, THREE.Vector3][] = [];
    for (const i of Array.from({ length: 25 }, (_, k) => k)) {
      const x = -d + (i % 5) * (2 * d / 4);
      const y = -d + Math.floor(i / 5) * (2 * d / 4);
      const v = fieldAt(x, y, 0, mode, cfg);
      if (v.lengthSq() < 1e-12) continue;
      v.normalize().multiplyScalar(.3);
      out.push([new THREE.Vector3(x, y, .12), new THREE.Vector3(x + v.x, y + v.y, .12 + v.z)]);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode + JSON.stringify(cfg)]);
  return (
    <group>
      {showVectors && vectors.map((v, i) => (
        <Line key={i} points={v} color={colors.vector} lineWidth={1.3} />
      ))}
    </group>
  );
}

/* ---------- 场源几何：间距 / 层数 / 半径 / 板间距全部参数化 ---------- */

function Source({ mode, cfg }: { mode: Mode; cfg: Cfg }) {
  if (mode === 'capacitor') {
    const c = cfg as CapCfg, h = c.gap / 2;
    return (
      <group>
        <mesh position={[-h, 0, 0]}>
          <boxGeometry args={[.16, 5.2, .55]} />
          <meshStandardMaterial color={colors.positive} />
        </mesh>
        <mesh position={[h, 0, 0]}>
          <boxGeometry args={[.16, 5.2, .55]} />
          <meshStandardMaterial color={colors.negative} />
        </mesh>
        <Text position={[-(h - .2), 2.75, .2]} fontSize={.2} color={colors.positive}>+Q</Text>
        <Text position={[h - .2, 2.75, .2]} fontSize={.2} color={colors.negative}>−Q</Text>
      </group>
    );
  }
  if (mode === 'coil') {
    const c = cfg as CoilCfg;
    return (
      <group>
        {Array.from({ length: c.layers }, (_, i) => (
          <mesh key={i} position={[0, 0, (i - (c.layers - 1) / 2) * .34]}>
            <torusGeometry args={[c.radius, .07, 10, 48]} />
            <meshStandardMaterial color={colors.coil} emissive={colors.coil} emissiveIntensity={c.strength * .12} />
          </mesh>
        ))}
      </group>
    );
  }
  const c = cfg as DipoleCfg, a = c.separation / 2;
  return (
    <group>
      <mesh position={[-a, 0, 0]}>
        <sphereGeometry args={[.28, 20, 16]} />
        <meshStandardMaterial color={colors.positive} emissive={colors.positive} emissiveIntensity={.25 + c.strength * .1} />
      </mesh>
      <mesh position={[a, 0, 0]}>
        <sphereGeometry args={[.28, 20, 16]} />
        <meshStandardMaterial color={colors.negative} emissive={colors.negative} emissiveIntensity={.25 + c.strength * .1} />
      </mesh>
      <Line points={[new THREE.Vector3(-a, 0, 0), new THREE.Vector3(a, 0, 0)]} color="#d8e4ef" lineWidth={2} />
    </group>
  );
}

function EMScene({ mode, cfg, showVectors }: { mode: Mode; cfg: Cfg; showVectors: boolean }) {
  return (
    <Canvas camera={{ position: [6, 4.5, 8], fov: 42 }}>
      <color attach="background" args={['#0b0b1c']} />
      <ambientLight intensity={1.2} />
      <pointLight position={[3, 4, 5]} color="#9a8cff" intensity={18} />
      <pointLight position={[-4, -2, 2]} color="#49d5e7" intensity={12} />
      {/* key 强制场型切换时整体重建，杜绝上一组几何 / 采样残留 */}
      <group key={mode} rotation={[.18, -.24, 0]}>
        <Source mode={mode} cfg={cfg} />
        <FieldLines mode={mode} cfg={cfg} />
        <Vectors mode={mode} cfg={cfg} showVectors={showVectors} />
        <mesh position={[0, 0, -.45]}>
          <planeGeometry args={[7, 6]} />
          <meshBasicMaterial color="#25234b" transparent opacity={.12} side={THREE.DoubleSide} />
        </mesh>
      </group>
      <OrbitControls makeDefault minDistance={5} maxDistance={14} />
    </Canvas>
  );
}

/* ---------- 面板控件 ---------- */

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function Slider({ label, value, min, max, step, digits, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; digits: number; unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="control">
      <span>{label}<b>{value.toFixed(digits)}{unit ?? ''}</b></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))} />
    </label>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>('dipole');
  // 每个场型独立保存参数；切换场型即恢复该模型上次使用的一组
  const [store, setStore] = useState<Store>(DEFAULTS);
  const [showVectors, setShowVectors] = useState(true);

  const cfg = store[mode];
  const update = (patch: Partial<DipoleCfg> & Partial<CoilCfg> & Partial<CapCfg>) =>
    setStore(s => ({ ...s, [mode]: { ...s[mode], ...patch } } as Store));
  const resetCurrent = () => setStore(s => ({ ...s, [mode]: DEFAULTS[mode] } as Store));

  const domain = sampleDomain(mode, cfg);
  const geometryMetric =
    mode === 'dipole' ? { label: '电荷间距 d', value: (cfg as DipoleCfg).separation.toFixed(2), note: '观察范围 ±' + (cfg as DipoleCfg).range.toFixed(1) }
    : mode === 'coil' ? { label: '线圈半径 R', value: (cfg as CoilCfg).radius.toFixed(2), note: `共 ${(cfg as CoilCfg).layers} 层绕线` }
    : { label: '板间距 D', value: (cfg as CapCfg).gap.toFixed(1), note: '两板净距' };
  const sourceScale =
    mode === 'dipole' ? `d = ${(cfg as DipoleCfg).separation.toFixed(2)}`
    : mode === 'coil' ? `R ${(cfg as CoilCfg).radius.toFixed(2)} · ${(cfg as CoilCfg).layers} 层`
    : `D = ${(cfg as CapCfg).gap.toFixed(1)}`;
  const direction = mode === 'coil' ? '右手螺旋定则' : '+ → −';

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">✦</div>
          <div>
            <span className="eyebrow">ELECTROMAGNETISM / FIELD EXPLORER</span>
            <h1>ElectroMag Lab <span>·</span> 电磁场与磁力线</h1>
          </div>
        </div>
        <div className="header-meta">
          <span className="dataset">MODEL 03 / {MODE_NAME[mode]}</span>
          <button onClick={() => setShowVectors(v => !v)}>{showVectors ? '隐藏场矢量' : '显示场矢量'}</button>
        </div>
      </header>
      <main className="layout">
        <section className="model-card">
          <div className="model-head">
            <div>
              <span className="eyebrow">3D FIELD MAPPING</span>
              <h2>{MODE_NAME[mode]} · Field Line Tracer</h2>
            </div>
            <div className="mode-switch">
              {MODES.map(item => (
                <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{MODE_SHORT[item]}</button>
              ))}
            </div>
          </div>
          <div className="canvas-wrap">
            <EMScene mode={mode} cfg={cfg} showVectors={showVectors} />
            <div className="viewer-legend">
              <span><i className="dot protein" />场线</span>
              <span><i className="dot ligand" />场源</span>
              <span><i className="dot interaction" />矢量采样</span>
            </div>
            <div className="viewer-hint">左键旋转 · 滚轮缩放 · 切换场源模型</div>
          </div>
        </section>
        <aside className="sidebar">
          <section className="panel intro-panel">
            <span className="eyebrow">FIELD EXPERIMENT</span>
            <h3>三维场线实验台</h3>
            <p>三种场型各自保存独立的形态参数：拖动滑杆即时改变场源几何与采样范围；切换模型会恢复其上次参数，重置仅作用于当前模型。</p>
            <div className="metrics">
              <Metric label="激励强度" value={cfg.strength.toFixed(1) + ' ×'} note="归一化系数" />
              <Metric label={geometryMetric.label} value={geometryMetric.value} note={geometryMetric.note} />
              <Metric label="采样域" value={`±${domain.toFixed(1)}`} note={`${LINE_COUNT[mode]} 束场线 · ${showVectors ? '25 矢量' : '矢量关'}`} />
            </div>
          </section>
          <section className="panel">
            <div className="panel-title">
              <div>
                <span className="eyebrow">SOURCE CONTROL</span>
                <h3>场源与强度</h3>
              </div>
              <button className="reset-btn" onClick={resetCurrent}>重置本模型</button>
            </div>
            <Slider label="激励强度" value={cfg.strength} min={.5} max={3} step={.1} digits={1} unit=" ×"
              onChange={v => update({ strength: v })} />
            <div className="param-divider"><span className="eyebrow">MODEL GEOMETRY · 按场型保存</span></div>
            {mode === 'dipole' && (
              <div className="param-group" key="dipole">
                <Slider label="电荷间距 d" value={(cfg as DipoleCfg).separation} min={.4} max={3} step={.05} digits={2}
                  onChange={v => update({ separation: v })} />
                <Slider label="观察范围" value={(cfg as DipoleCfg).range} min={2.4} max={6} step={.1} digits={1}
                  onChange={v => update({ range: v })} />
              </div>
            )}
            {mode === 'coil' && (
              <div className="param-group" key="coil">
                <Slider label="线圈层数" value={(cfg as CoilCfg).layers} min={3} max={12} step={1} digits={0} unit=" 层"
                  onChange={v => update({ layers: v })} />
                <Slider label="线圈半径 R" value={(cfg as CoilCfg).radius} min={1} max={2.4} step={.05} digits={2}
                  onChange={v => update({ radius: v })} />
              </div>
            )}
            {mode === 'capacitor' && (
              <div className="param-group" key="capacitor">
                <Slider label="板间距 D" value={(cfg as CapCfg).gap} min={2} max={8} step={.1} digits={1}
                  onChange={v => update({ gap: v })} />
              </div>
            )}
            <div className="source-list">
              {MODES.map(item => (
                <button key={item} className={mode === item ? 'source active' : 'source'} onClick={() => setMode(item)}>
                  <span className={`source-dot ${item}`} />
                  {item === 'dipole' ? '电偶极子' : item === 'coil' ? '通电螺线管' : '平行板电容器'}
                  <em>{item === 'coil' ? 'B' : 'E'}</em>
                </button>
              ))}
            </div>
          </section>
          <section className="panel">
            <span className="eyebrow">FIELD READOUT</span>
            <div className="readout">
              <div><span>当前场型</span><b>{FIELD_KIND[mode]}</b></div>
              <div><span>方向约定</span><b>{direction}</b></div>
              <div><span>场源尺度</span><b>{sourceScale}</b></div>
              <div><span>采样与场线</span><b>±{domain.toFixed(1)} · {LINE_COUNT[mode]} 束 · 教学级采样</b></div>
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}

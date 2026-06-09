import React, { useState, useCallback, useRef, useEffect } from 'react'
import { Upload, X, Film, CheckCircle2, AlertCircle, Loader2, Play, Database, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

const API = ''
const TASK_STORAGE_KEY = 'svq_pipeline_task'
const POLL_TIMEOUT_MS = 30 * 60 * 1000  // 30 min

// ── helpers ───────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 ** 2).toFixed(1)} MB`
}
function fmtMs(ms) {
  if (!ms) return '—'
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`
  return `${(ms / 60000).toFixed(1)} min`
}

// ── Card shell ────────────────────────────────────────────────────────────────
function Card({ icon, title, badge, children }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-5">
      <div className="bg-slate-50 border-b border-slate-200 px-4 py-3 flex items-center gap-2">
        {icon && <span className="text-base">{icon}</span>}
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        {badge && (
          <span className="ml-auto text-xs bg-blue-100 text-blue-700 font-semibold px-2 py-0.5 rounded-full">{badge}</span>
        )}
      </div>
      <div className="p-4">{children}</div>
    </div>
  )
}

// ── Section divider ───────────────────────────────────────────────────────────
function Section({ label }) {
  return (
    <div className="flex items-center gap-2 mb-3 mt-1">
      <span className="text-xs font-bold uppercase tracking-widest text-slate-400">{label}</span>
      <div className="flex-1 h-px bg-slate-200" />
    </div>
  )
}

// ── Drop zone ─────────────────────────────────────────────────────────────────
function DropZone({ onFiles, multiple = true }) {
  const [dragging, setDragging] = useState(false)
  const ref = useRef()
  const handle = useCallback((files) => {
    const vids = [...files].filter(f => f.type.startsWith('video/') || /\.(mp4|avi|mov|mkv)$/i.test(f.name))
    if (vids.length) onFiles(multiple ? vids : [vids[0]])
  }, [onFiles, multiple])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files) }}
      onClick={() => ref.current.click()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all mb-4
        ${dragging ? 'border-blue-400 bg-blue-50' : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50'}`}
    >
      <div className="text-3xl mb-2">☁️</div>
      <p className="text-sm font-semibold text-slate-600 mb-1">
        {multiple ? 'Drag & drop video files here' : 'Drag & drop a video file here'}
      </p>
      <p className="text-xs text-slate-400">
        or <span className="text-blue-500 font-semibold underline">browse files</span> · MP4, AVI, MKV, MOV
      </p>
      <input ref={ref} type="file" multiple={multiple} accept="video/*,.avi,.mkv" className="hidden"
        onChange={e => handle(e.target.files)} />
    </div>
  )
}

// ── Video file row ────────────────────────────────────────────────────────────
function VideoRow({ item, onChange, onRemove, showCamId = true }) {
  return (
    <div className="flex items-center gap-3 py-2.5 px-3 bg-slate-50 border border-slate-200 rounded-lg mb-2">
      <div className="w-11 h-8 rounded bg-slate-800 flex items-center justify-center flex-shrink-0">
        <Film className="w-3.5 h-3.5 text-slate-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-slate-700 truncate">{item.file.name}</p>
        <p className="text-xs text-slate-400">{fmtBytes(item.file.size)}</p>
      </div>
      {showCamId && (
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className="text-xs text-slate-500">Camera ID</span>
          <input value={item.cameraId} onChange={e => onChange({ ...item, cameraId: e.target.value })}
            placeholder="CAM-01"
            className="w-20 border border-slate-300 rounded px-2 py-1 text-xs text-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-400" />
        </div>
      )}
      <button onClick={onRemove} className="text-slate-400 hover:text-red-500 transition-colors">
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

// ── Object class checkbox grid ────────────────────────────────────────────────
const ALL_CLASSES = [
  { key: 'person', emoji: '🧍', label: 'person' },
  { key: 'car', emoji: '🚗', label: 'car' },
  { key: 'bus', emoji: '🚌', label: 'bus' },
  { key: 'truck', emoji: '🚛', label: 'truck' },
  { key: 'motorcycle', emoji: '🏍️', label: 'motorcycle' },
  { key: 'bicycle', emoji: '🚲', label: 'bicycle' },
]

function ClassGrid({ selected, onChange }) {
  const toggle = key => onChange(
    selected.includes(key) ? selected.filter(k => k !== key) : [...selected, key]
  )
  return (
    <div className="grid grid-cols-3 gap-2">
      {ALL_CLASSES.map(({ key, emoji, label }) => {
        const on = selected.includes(key)
        return (
          <label key={key}
            className={`flex items-center gap-1.5 px-2.5 py-2 rounded-lg border cursor-pointer text-xs transition-all
              ${on ? 'border-blue-400 bg-blue-50 text-blue-700 font-semibold' : 'border-slate-200 bg-slate-50 text-slate-500 hover:border-slate-300'}`}
          >
            <input type="checkbox" checked={on} onChange={() => toggle(key)} className="accent-blue-500" />
            {emoji} {label}
          </label>
        )
      })}
    </div>
  )
}

// ── Range input ───────────────────────────────────────────────────────────────
function RangeField({ label, value, min, max, step, onChange, unit = '' }) {
  return (
    <div>
      <label className="block text-xs font-semibold text-slate-500 mb-1">{label}</label>
      <div className="flex items-center gap-2">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 accent-blue-500" />
        <span className="text-xs font-bold text-blue-600 min-w-[36px] text-right">{value}{unit}</span>
      </div>
    </div>
  )
}

// ── Reset DB ──────────────────────────────────────────────────────────────────
function ResetCard({ namespace }) {
  const [confirm, setConfirm] = useState(false)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState(null)  // {ok, text}

  const doReset = async () => {
    setLoading(true); setMsg(null)
    try {
      const body = { namespace }
      const r = await fetch(`${API}/api/v1/database/reset`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || r.statusText)
      setMsg({ ok: true, text: data.message || 'Cleared' })
    } catch (err) {
      setMsg({ ok: false, text: err.message })
    } finally {
      setLoading(false); setConfirm(false)
    }
  }

  return (
    <Card title="Database" icon="🗄️">
      <p className="text-xs text-slate-500 mb-3">
        Drop all SQLite rows and Chroma vectors for namespace <span className="font-mono font-semibold">{namespace || '—'}</span>.
      </p>
      {!confirm ? (
        <button onClick={() => setConfirm(true)}
          className="w-full border border-red-300 text-red-600 text-xs font-semibold py-2 rounded-lg
                     hover:bg-red-50 transition-colors">
          Reset Database
        </button>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-red-600 font-medium">This cannot be undone. Confirm?</p>
          <div className="flex gap-2">
            <button onClick={doReset} disabled={loading}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold py-1.5 rounded-lg
                         disabled:opacity-50 transition-colors">
              {loading ? 'Clearing…' : 'Yes, reset'}
            </button>
            <button onClick={() => setConfirm(false)}
              className="flex-1 border border-slate-300 text-slate-600 text-xs font-semibold py-1.5 rounded-lg
                         hover:bg-slate-50 transition-colors">
              Cancel
            </button>
          </div>
        </div>
      )}
      {msg && (
        <p className={`text-xs mt-2 font-medium ${msg.ok ? 'text-green-600' : 'text-red-600'}`}>{msg.text}</p>
      )}
    </Card>
  )
}

// ── Run Summary sidebar ───────────────────────────────────────────────────────
function RunSummary({ mode, files, cfg, isRunning, uploading, onRun, canRun, taskStatus, onStopWaiting }) {
  const count = mode === 'single' ? (files ? 1 : 0) : files.length

  return (
    <div className="w-80 flex-shrink-0 space-y-4">
      {/* Stats */}
      <Card title="Run Summary">
        <div className="grid grid-cols-3 gap-2 mb-4">
          {[
            { val: count, lbl: 'Videos' },
            { val: mode === 'single' ? 1 : count, lbl: 'Cameras' },
            { val: cfg.namespace || '—', lbl: 'Namespace' },
          ].map(({ val, lbl }) => (
            <div key={lbl} className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-center">
              <div className="text-lg font-bold text-blue-600 truncate">{val}</div>
              <div className="text-xs text-slate-400 mt-0.5">{lbl}</div>
            </div>
          ))}
        </div>

        <div className="space-y-0 divide-y divide-slate-100 mb-4">
          {[
            ['Mode', mode === 'single' ? 'Single Camera' : 'Multi Camera'],
            ['Conf', cfg.conf],
            ['IoU', cfg.iou],
            ...(mode === 'single' ? [
              ['Clip refine', cfg.clipRefine ? '✓ On' : 'Off'],
              ['Appearance', cfg.appearanceRefine ? '✓ On' : 'Off'],
            ] : [
              ['Re-ID device', cfg.reidDevice],
              ['LLM refine', cfg.refine ? '✓ On' : 'Off'],
            ]),
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between py-1.5">
              <span className="text-xs text-slate-500">{k}</span>
              <span className={`text-xs font-semibold ${String(v).startsWith('✓') ? 'text-green-600' : 'text-slate-700'}`}>{v}</span>
            </div>
          ))}
        </div>

        <button onClick={onRun} disabled={!canRun}
          className="w-full bg-blue-700 hover:bg-blue-800 text-white font-semibold text-sm py-2.5 rounded-lg
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2">
          {uploading ? <><Loader2 className="w-4 h-4 animate-spin" />Uploading…</>
            : isRunning ? <><Loader2 className="w-4 h-4 animate-spin" />Running…</>
            : <><Play className="w-4 h-4" />Start Pipeline</>}
        </button>
      </Card>

      {/* Reset */}
      <ResetCard namespace={cfg.namespace} />

      {/* Existing databases */}
      <ExistingDBCard onActivated={() => {}} />

      {/* Status */}
      {taskStatus && (
        <Card title="Pipeline Status" icon={
          taskStatus.status === 'done' ? '✅' : taskStatus.status === 'error' ? '❌' : '⏳'
        }>
          {taskStatus.status === 'running' && (
            <RunningStages
              stage={taskStatus.stage}
              startedAt={taskStatus.started_at}
              mode={mode}
              onStop={onStopWaiting}
            />
          )}
          {taskStatus.status === 'error' && (
            <p className="text-xs text-red-600">{taskStatus.error}</p>
          )}
          {taskStatus.status === 'done' && <DoneStats result={taskStatus.result} />}
        </Card>
      )}
    </div>
  )
}

// ── Pipeline running stages ───────────────────────────────────────────────────
const PIPELINE_STAGES = [
  { key: 'detection', label: 'Detection & Tracking' },
  { key: 'indexing',  label: 'Building Index' },
]

function RunningStages({ stage, startedAt, mode, onStop }) {
  const [elapsed, setElapsed] = useState(
    startedAt ? Math.floor(Date.now() / 1000 - startedAt) : 0
  )
  useEffect(() => {
    const t = setInterval(() =>
      setElapsed(startedAt ? Math.floor(Date.now() / 1000 - startedAt) : 0), 1000)
    return () => clearInterval(t)
  }, [startedAt])

  const activeIdx = Math.max(0, PIPELINE_STAGES.findIndex(s => s.key === (stage || 'detection')))

  return (
    <div className="space-y-3">
      {PIPELINE_STAGES.map((s, i) => {
        const isDone   = i < activeIdx
        const isActive = i === activeIdx
        return (
          <div key={s.key}>
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span className={isDone ? 'line-through text-slate-300' : ''}>{s.label}</span>
              <span className={
                isDone   ? 'text-green-500 font-medium' :
                isActive ? 'text-blue-500 font-medium'  : 'text-slate-300'
              }>
                {isDone ? '✓ done' : isActive ? 'processing…' : 'waiting'}
              </span>
            </div>
            <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all duration-700
                ${isDone ? 'w-full bg-green-400' : isActive ? 'w-2/3 bg-blue-500 animate-pulse' : 'w-0'}`} />
            </div>
          </div>
        )
      })}
      <p className="text-xs text-slate-400 mt-1">
        {elapsed > 0 ? `Running ${elapsed}s… · ` : ''}
        {mode === 'single' ? 'Usually 1–5 min' : 'May take 10–60 min with Re-ID'}
      </p>
      <button onClick={onStop}
        className="text-xs text-slate-400 hover:text-red-500 underline mt-1">
        Stop waiting
      </button>
    </div>
  )
}

function DoneStats({ result }) {
  const [open, setOpen] = useState(false)
  if (!result) return <p className="text-xs text-slate-400">Finalising…</p>
  const rows = [
    ['SQLite rows', result.sqlite_rows],
    ['Chroma child', result.chroma_child_records],
    ['Seed events', result.seed_event_count ?? result.seed_file_count],
    ['Elapsed', fmtMs(result.elapsed_ms)],
  ].filter(([, v]) => v != null)

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-green-600 text-sm font-semibold">
        <CheckCircle2 className="w-4 h-4" /> Database ready
      </div>
      <p className="text-xs text-slate-500 font-mono truncate">{result.namespace}</p>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between text-xs">
          <span className="text-slate-500">{k}</span>
          <span className="font-semibold text-slate-700">{v}</span>
        </div>
      ))}
      <button onClick={() => setOpen(v => !v)} className="text-xs text-blue-500 hover:underline flex items-center gap-1">
        {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        {open ? 'Hide paths' : 'Show paths'}
      </button>
      {open && (
        <div className="text-xs text-slate-400 space-y-1 break-all">
          <p>{result.sqlite_path}</p>
          <p>{result.chroma_path}</p>
        </div>
      )}
    </div>
  )
}

// ── Existing databases quick-select ──────────────────────────────────────────
function ExistingDBCard({ onActivated }) {
  const [dbs, setDbs] = useState([])
  const [selected, setSelected] = useState('')
  const [loading, setLoading] = useState(true)
  const [activating, setActivating] = useState(null)

  const refresh = useCallback(() => {
    setLoading(true)
    fetch(`${API}/api/v1/databases`)
      .then(r => r.json())
      .then(data => { setDbs(data.databases || []); setSelected(data.selected || '') })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const activate = async (db) => {
    setActivating(db.namespace)
    try {
      const r = await fetch(`${API}/api/v1/database/select`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ namespace: db.namespace, sqlite_path: db.sqlite_path, chroma_path: db.chroma_path }),
      })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
      setSelected(db.namespace)
      onActivated && onActivated(db)
    } catch (e) { alert(`Failed to load: ${e.message}`) }
    finally { setActivating(null) }
  }

  if (!loading && !dbs.length) return null

  return (
    <Card title="Existing Databases" icon="📂">
      {loading ? (
        <div className="flex items-center gap-2 text-xs text-slate-400"><Loader2 className="w-3 h-3 animate-spin" />Loading…</div>
      ) : (
        <div className="space-y-2">
          {dbs.map(db => {
            const isActive = selected === db.namespace
            const isBusy = activating === db.namespace
            return (
              <div key={db.namespace}
                className={`flex items-center gap-2 p-2.5 rounded-lg border text-xs transition-all
                  ${isActive ? 'border-green-400 bg-green-50' : 'border-slate-200 bg-slate-50 hover:border-slate-300'}`}>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-slate-700 font-mono truncate">{db.namespace}</p>
                  <p className="text-slate-400 mt-0.5">
                    {db.row_count} rows · {db.size_kb} KB
                    {db.cameras.length > 0 && <> · {db.cameras.slice(0, 3).join(', ')}{db.cameras.length > 3 ? '…' : ''}</>}
                  </p>
                </div>
                {isActive ? (
                  <span className="flex items-center gap-1 text-green-600 font-semibold flex-shrink-0">
                    <CheckCircle2 className="w-3.5 h-3.5" />Active
                  </span>
                ) : (
                  <button onClick={() => activate(db)} disabled={!!activating}
                    className="flex-shrink-0 text-xs bg-blue-600 hover:bg-blue-700 text-white px-2.5 py-1 rounded-md
                               disabled:opacity-50 transition-colors flex items-center gap-1">
                    {isBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
                    Load
                  </button>
                )}
              </div>
            )
          })}
          <button onClick={refresh} className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 mt-1">
            <RefreshCw className="w-3 h-3" />Refresh list
          </button>
        </div>
      )}
    </Card>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UploadPage() {
  const [mode, setMode] = useState('single')

  // single state
  const [singleFile, setSingleFile] = useState(null)
  const [sCfg, setSCfg] = useState({
    namespace: 'mevid', cameraId: '', conf: 0.35, iou: 0.35,
    clipRefine: false, clipRefineModel: 'qwen3-vl-flash',
    appearanceRefine: false, sceneProfile: false,
    classes: ['person', 'car'],
  })

  // multi state
  const [multiItems, setMultiItems] = useState([])
  const [mCfg, setMCfg] = useState({
    namespace: 'mevid', conf: 0.4, iou: 0.4, reidDevice: 'cpu',
    refine: false, classes: ['person', 'car'],
  })

  const [uploading, setUploading] = useState(false)
  const [taskId, setTaskId] = useState(null)
  const [taskStatus, setTaskStatus] = useState(null)
  const pollRef = useRef(null)
  const pollStartRef = useRef(null)

  const cfg = mode === 'single' ? sCfg : mCfg

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    localStorage.removeItem(TASK_STORAGE_KEY)
    setTaskId(null)
  }, [])

  const stopWaiting = useCallback(() => {
    stopPolling()
    setTaskStatus(null)
  }, [stopPolling])

  // ── Restore task from localStorage on mount ──────────────────────────────
  useEffect(() => {
    try {
      const saved = localStorage.getItem(TASK_STORAGE_KEY)
      if (saved) {
        const { taskId: tid, startedAt } = JSON.parse(saved)
        if (tid && Date.now() - startedAt < POLL_TIMEOUT_MS) {
          setTaskId(tid)
          pollStartRef.current = startedAt
          setTaskStatus({ status: 'running', result: null, error: null })
        } else {
          localStorage.removeItem(TASK_STORAGE_KEY)
        }
      }
    } catch (_) {}
  }, [])

  // ── Polling with timeout ─────────────────────────────────────────────────
  useEffect(() => {
    if (!taskId) return
    if (pollRef.current) clearInterval(pollRef.current)
    const poll = async () => {
      // 30-min hard timeout
      if (pollStartRef.current && Date.now() - pollStartRef.current > POLL_TIMEOUT_MS) {
        setTaskStatus({ status: 'error', result: null, error: 'Pipeline timed out after 30 minutes. Check backend logs.' })
        stopPolling()
        return
      }
      try {
        const r = await fetch(`${API}/api/v1/pipeline/status/${taskId}`)
        const data = await r.json().catch(() => ({}))
        if (!r.ok) throw new Error(data.detail || `Task status failed (${r.status})`)
        setTaskStatus(data)
        if (data.status === 'done' || data.status === 'error') {
          stopPolling()
        }
      } catch (err) {
        setTaskStatus({ status: 'error', result: null,
          error: `${err.message}. The backend may have restarted — start the pipeline again.` })
        stopPolling()
      }
    }
    poll()
    pollRef.current = setInterval(poll, 2500)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [taskId, stopPolling])

  const switchMode = m => { setMode(m); stopWaiting() }

  // run single
  const runSingle = async () => {
    if (!singleFile) return
    stopPolling()
    setUploading(true); setTaskStatus(null); setTaskId(null)
    try {
      const form = new FormData(); form.append('files', singleFile)
      const upRes = await fetch(`${API}/api/v1/upload/videos`, { method: 'POST', body: form })
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`)
      const upData = await upRes.json()
      const pRes = await fetch(`${API}/api/v1/pipeline/start/single`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: upData.files[0].path, namespace: sCfg.namespace,
          camera_id: sCfg.cameraId || null, conf: sCfg.conf, iou: sCfg.iou,
          target_classes: sCfg.classes,
          clip_refine: sCfg.clipRefine, clip_refine_model: sCfg.clipRefineModel,
          appearance_refine: sCfg.appearanceRefine, scene_profile: sCfg.sceneProfile,
          reset_existing: true,
        }),
      })
      if (!pRes.ok) throw new Error(`Pipeline failed: ${pRes.statusText}`)
      const pData = await pRes.json()
      const now = Date.now()
      pollStartRef.current = now
      localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify({ taskId: pData.task_id, startedAt: now }))
      setTaskId(pData.task_id); setTaskStatus({ status: 'running', result: null, error: null })
    } catch (err) {
      setTaskStatus({ status: 'error', result: null, error: err.message })
    } finally { setUploading(false) }
  }

  // run multi
  const runMulti = async () => {
    if (!multiItems.length) return
    stopPolling()
    setUploading(true); setTaskStatus(null); setTaskId(null)
    try {
      const form = new FormData(); multiItems.forEach(it => form.append('files', it.file))
      const upRes = await fetch(`${API}/api/v1/upload/videos`, { method: 'POST', body: form })
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`)
      const upData = await upRes.json()
      const camera_videos = {}
      upData.files.forEach((f, i) => { camera_videos[multiItems[i].cameraId] = f.path })
      const pRes = await fetch(`${API}/api/v1/pipeline/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_videos, namespace: mCfg.namespace, conf: mCfg.conf, iou: mCfg.iou,
          target_classes: mCfg.classes,
          reid_device: mCfg.reidDevice, refine: mCfg.refine, reset_existing: true,
        }),
      })
      if (!pRes.ok) throw new Error(`Pipeline failed: ${pRes.statusText}`)
      const pData = await pRes.json()
      const now = Date.now()
      pollStartRef.current = now
      localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify({ taskId: pData.task_id, startedAt: now }))
      setTaskId(pData.task_id); setTaskStatus({ status: 'running', result: null, error: null })
    } catch (err) {
      setTaskStatus({ status: 'error', result: null, error: err.message })
    } finally { setUploading(false) }
  }

  const addMulti = useCallback(files => {
    setMultiItems(prev => {
      const ex = new Set(prev.map(i => i.file.name))
      return [...prev, ...files.filter(f => !ex.has(f.name)).map(f => ({
        file: f,
        cameraId: f.name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9]/g, '_').slice(0, 8).toUpperCase(),
      }))]
    })
  }, [])

  const isRunning = taskStatus?.status === 'running' || uploading
  const canRun = mode === 'single'
    ? !!singleFile && !isRunning
    : multiItems.length > 0 && !isRunning && multiItems.every(i => i.cameraId.trim())

  // ── render ──
  return (
    <div className="max-w-[1100px] mx-auto px-6 py-7 pb-12">
      <h1 className="text-xl font-bold text-slate-800 mb-1">Video Ingestion Pipeline</h1>
      <p className="text-sm text-slate-500 mb-6">Upload surveillance footage and configure processing parameters before indexing.</p>

      <div className="flex gap-5 items-start">

        {/* ── Left column ── */}
        <div className="flex-1 min-w-0">

          {/* Mode selector */}
          <div className="flex gap-3 mb-5">
            {[
              { key: 'single', emoji: '📷', label: 'Single Camera', desc: 'One video, no Re-ID. Fast.' },
              { key: 'multi', emoji: '🎥', label: 'Multi Camera', desc: 'Cross-view Re-ID matching.' },
            ].map(({ key, emoji, label, desc }) => (
              <button key={key} onClick={() => switchMode(key)}
                className={`flex-1 text-left px-4 py-3 rounded-xl border text-sm transition-all
                  ${mode === key
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-slate-200 bg-white text-slate-500 hover:border-slate-300'}`}>
                <div className="font-semibold mb-0.5">{emoji} {label}</div>
                <div className="text-xs opacity-70">{desc}</div>
              </button>
            ))}
          </div>

          {/* ── Single camera ── */}
          {mode === 'single' && (
            <>
              <Card icon="📹" title="Video Upload">
                {singleFile ? (
                  <VideoRow item={{ file: singleFile, cameraId: sCfg.cameraId }}
                    onChange={u => setSCfg(c => ({ ...c, cameraId: u.cameraId }))}
                    onRemove={() => setSingleFile(null)}
                    showCamId />
                ) : (
                  <DropZone onFiles={fs => setSingleFile(fs[0])} multiple={false} />
                )}
              </Card>

              <Card icon="🏷️" title="Camera Identity">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-1">Namespace</label>
                    <input value={sCfg.namespace} onChange={e => setSCfg(c => ({ ...c, namespace: e.target.value }))}
                      className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400" />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-1">Camera ID (optional)</label>
                    <input value={sCfg.cameraId} onChange={e => setSCfg(c => ({ ...c, cameraId: e.target.value }))}
                      placeholder="auto-infer from filename"
                      className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400" />
                  </div>
                </div>
              </Card>

              <Card icon="🎯" title="Detection & Tracking">
                <Section label="Thresholds" />
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <RangeField label="Detection Confidence" value={sCfg.conf} min={0.1} max={0.9} step={0.05}
                    onChange={v => setSCfg(c => ({ ...c, conf: v }))} />
                  <RangeField label="Tracking IoU" value={sCfg.iou} min={0.1} max={0.9} step={0.05}
                    onChange={v => setSCfg(c => ({ ...c, iou: v }))} />
                </div>
                <Section label="Target Object Classes" />
                <ClassGrid selected={sCfg.classes} onChange={v => setSCfg(c => ({ ...c, classes: v }))} />
              </Card>

              <Card icon="✨" title="Semantic Refinement">
                <Section label="LLM / VLM Enrichment" />
                <div className="space-y-3">
                  {[
                    { key: 'clipRefine', label: 'Clip Refine (LLM event text)', desc: 'LLM watches sampled frames and generates event descriptions.' },
                    { key: 'appearanceRefine', label: 'Appearance Refine (VLM)', desc: 'VLM analyzes person crop images for clothing color and description.' },
                    { key: 'sceneProfile', label: 'Scene Profile', desc: 'Generate scene zone context for each camera.' },
                  ].map(({ key, label, desc }) => (
                    <label key={key}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all
                        ${sCfg[key] ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-slate-50 hover:border-slate-300'}`}>
                      <input type="checkbox" checked={sCfg[key]} onChange={e => setSCfg(c => ({ ...c, [key]: e.target.checked }))}
                        className="accent-blue-500 mt-0.5" />
                      <div>
                        <p className={`text-sm font-semibold ${sCfg[key] ? 'text-blue-700' : 'text-slate-600'}`}>{label}</p>
                        <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                      </div>
                    </label>
                  ))}
                  {sCfg.clipRefine && (
                    <div className="pl-7">
                      <label className="block text-xs font-semibold text-slate-500 mb-1">Clip Refine Model</label>
                      <select value={sCfg.clipRefineModel}
                        onChange={e => setSCfg(c => ({ ...c, clipRefineModel: e.target.value }))}
                        className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400">
                        <option value="qwen3-vl-flash">qwen3-vl-flash (vision, recommended)</option>
                        <option value="qwen3-vl-plus">qwen3-vl-plus (vision, higher quality)</option>
                        <option value="qwen3.6-plus">qwen3.6-plus</option>
                        <option value="gpt-4o-mini">gpt-4o-mini</option>
                        <option value="gpt-4o">gpt-4o</option>
                      </select>
                    </div>
                  )}
                </div>
              </Card>
            </>
          )}

          {/* ── Multi camera ── */}
          {mode === 'multi' && (
            <>
              <Card icon="📹" title="Video Upload" badge={multiItems.length ? `${multiItems.length} queued` : undefined}>
                <DropZone onFiles={addMulti} multiple />
                {multiItems.map((item, idx) => (
                  <VideoRow key={idx} item={item}
                    onChange={u => setMultiItems(p => p.map((it, i) => i === idx ? u : it))}
                    onRemove={() => setMultiItems(p => p.filter((_, i) => i !== idx))}
                    showCamId />
                ))}
              </Card>

              <Card icon="🏷️" title="Namespace">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-1">Namespace</label>
                    <input value={mCfg.namespace} onChange={e => setMCfg(c => ({ ...c, namespace: e.target.value }))}
                      className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400" />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-1">Re-ID Device</label>
                    <select value={mCfg.reidDevice} onChange={e => setMCfg(c => ({ ...c, reidDevice: e.target.value }))}
                      className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400">
                      <option value="cpu">CPU</option>
                      <option value="cuda">CUDA (GPU)</option>
                    </select>
                  </div>
                </div>
              </Card>

              <Card icon="🎯" title="Detection & Tracking">
                <Section label="Thresholds" />
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <RangeField label="Detection Confidence" value={mCfg.conf} min={0.1} max={0.9} step={0.05}
                    onChange={v => setMCfg(c => ({ ...c, conf: v }))} />
                  <RangeField label="Tracking IoU" value={mCfg.iou} min={0.1} max={0.9} step={0.05}
                    onChange={v => setMCfg(c => ({ ...c, iou: v }))} />
                </div>
                <Section label="Target Object Classes" />
                <ClassGrid selected={mCfg.classes} onChange={v => setMCfg(c => ({ ...c, classes: v }))} />
              </Card>

              <Card icon="✨" title="LLM Event Refinement">
                <label className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all
                  ${mCfg.refine ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-slate-50'}`}>
                  <input type="checkbox" checked={mCfg.refine} onChange={e => setMCfg(c => ({ ...c, refine: e.target.checked }))}
                    className="accent-blue-500 mt-0.5" />
                  <div>
                    <p className={`text-sm font-semibold ${mCfg.refine ? 'text-blue-700' : 'text-slate-600'}`}>LLM Event Refinement</p>
                    <p className="text-xs text-slate-400 mt-0.5">Run VLM on each event clip to generate descriptive event text.</p>
                  </div>
                </label>
              </Card>
            </>
          )}
        </div>

        {/* ── Right sidebar ── */}
        <RunSummary
          mode={mode}
          files={mode === 'single' ? singleFile : multiItems}
          cfg={cfg}
          isRunning={isRunning}
          uploading={uploading}
          onRun={mode === 'single' ? runSingle : runMulti}
          canRun={canRun}
          taskStatus={taskStatus}
          onStopWaiting={stopWaiting}
        />
      </div>
    </div>
  )
}

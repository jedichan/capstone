import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Search, Loader2, AlertCircle, Clock, MessageSquare, Send, Database, CheckCircle2, RefreshCw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API = ''

function DatabaseSelector() {
  const [dbs, setDbs] = useState([])
  const [activeNs, setActiveNs] = useState('')   // currently loaded in agent
  const [chosenNs, setChosenNs] = useState('')   // user's dropdown choice
  const [listLoading, setListLoading] = useState(true)
  const [activating, setActivating] = useState(false)
  const [status, setStatus] = useState(null)

  const loadList = useCallback(() => {
    setListLoading(true)
    fetch(`${API}/api/v1/databases`)
      .then(r => r.json())
      .then(data => {
        const list = data.databases || []
        setDbs(list)
        const cur = data.selected || ''
        setActiveNs(cur)
        // Pre-select the already-active db; fall back to first in list
        setChosenNs(prev => prev || cur || (list[0]?.namespace ?? ''))
      })
      .catch(() => {})
      .finally(() => setListLoading(false))
  }, [])

  useEffect(() => { loadList() }, [loadList])

  const selectDb = async () => {
    const db = dbs.find(d => d.namespace === chosenNs)
    if (!db) return
    setActivating(true)
    setStatus(null)
    try {
      const res = await fetch(`${API}/api/v1/database/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ namespace: db.namespace, sqlite_path: db.sqlite_path, chroma_path: db.chroma_path }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || res.statusText)
      setActiveNs(db.namespace)
      setStatus({ ok: true, text: `Loaded ${data.namespace}` })
    } catch (err) {
      setStatus({ ok: false, text: err.message })
    } finally {
      setActivating(false)
    }
  }

  const isAlreadyActive = chosenNs === activeNs

  return (
    <div className="border-b border-slate-200 bg-white px-6 py-3">
      <div className="max-w-4xl mx-auto flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
          <Database className="w-4 h-4 text-blue-500" />
          Database
        </div>

        {listLoading ? (
          <span className="flex items-center gap-1.5 text-xs text-slate-400">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading databases…
          </span>
        ) : dbs.length === 0 ? (
          <span className="text-xs text-slate-400">No databases found — run the pipeline first.</span>
        ) : (
          <select
            value={chosenNs}
            onChange={e => { setChosenNs(e.target.value); setStatus(null) }}
            className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm text-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-400"
          >
            {dbs.map(d => (
              <option key={d.namespace} value={d.namespace}>
                {d.namespace}  ({d.row_count} rows{d.cameras.length ? ' · ' + d.cameras.slice(0, 3).join(', ') + (d.cameras.length > 3 ? '…' : '') : ''})
              </option>
            ))}
          </select>
        )}

        {!listLoading && dbs.length > 0 && (
          <button
            onClick={selectDb}
            disabled={activating || isAlreadyActive}
            className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-3 py-1.5 rounded-lg
                       disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            {activating
              ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Loading…</>
              : isAlreadyActive
              ? <><CheckCircle2 className="w-3.5 h-3.5" />Active</>
              : 'Load'}
          </button>
        )}

        <button onClick={loadList} className="text-slate-400 hover:text-slate-600" title="Refresh list">
          <RefreshCw className="w-3.5 h-3.5" />
        </button>

        {status && (
          <span className={`text-xs font-medium ${status.ok ? 'text-green-600' : 'text-red-600'}`}>
            {status.text}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Markdown prose styles ─────────────────────────────────────────────────────
const mdComponents = {
  p:          ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  strong:     ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
  em:         ({ children }) => <em className="italic">{children}</em>,
  h1:         ({ children }) => <h1 className="text-base font-bold text-slate-800 mt-3 mb-1">{children}</h1>,
  h2:         ({ children }) => <h2 className="text-sm font-bold text-slate-800 mt-3 mb-1">{children}</h2>,
  h3:         ({ children }) => <h3 className="text-sm font-semibold text-slate-700 mt-2 mb-1">{children}</h3>,
  ul:         ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-2 pl-1">{children}</ul>,
  ol:         ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-2 pl-1">{children}</ol>,
  li:         ({ children }) => <li className="text-sm text-slate-800">{children}</li>,
  code:       ({ inline, children }) => inline
    ? <code className="bg-slate-100 text-blue-700 text-xs px-1 py-0.5 rounded font-mono">{children}</code>
    : <pre className="bg-slate-100 text-xs rounded p-2 overflow-x-auto font-mono mb-2"><code>{children}</code></pre>,
  blockquote: ({ children }) => <blockquote className="border-l-2 border-blue-300 pl-3 text-slate-500 italic mb-2">{children}</blockquote>,
  hr:         () => <hr className="border-slate-200 my-2" />,
  a:          ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">{children}</a>,
  table:      ({ children }) => <div className="overflow-x-auto mb-2"><table className="text-xs border-collapse w-full">{children}</table></div>,
  th:         ({ children }) => <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-semibold">{children}</th>,
  td:         ({ children }) => <td className="border border-slate-200 px-2 py-1">{children}</td>,
}

// ── Result block ──────────────────────────────────────────────────────────────
function ResultBlock({ result }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-3">
      <div className="flex gap-3">
        <MessageSquare className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-slate-800 min-w-0 flex-1">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {result.answer}
          </ReactMarkdown>
        </div>
      </div>
      <div className="flex items-center gap-4 pt-1 text-xs text-slate-400 border-t border-slate-100">
        {result.elapsed_ms && <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{(result.elapsed_ms / 1000).toFixed(1)}s</span>}
        {result.answer_type && <span>{result.answer_type}</span>}
        {result.node_trace?.length > 0 && <span className="truncate">{result.node_trace.join(' → ')}</span>}
      </div>
    </div>
  )
}

// ── Chat history item ─────────────────────────────────────────────────────────
function HistoryItem({ item }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <div className="max-w-lg bg-blue-600 text-white rounded-2xl rounded-tr-md px-4 py-2.5">
          <p className="text-sm">{item.query}</p>
        </div>
      </div>
      {item.loading ? (
        <div className="flex items-center gap-2 text-slate-400 text-sm px-1">
          <Loader2 className="w-4 h-4 animate-spin" /> Thinking…
        </div>
      ) : item.error ? (
        <div className="flex items-center gap-2 text-red-500 text-sm px-1">
          <AlertCircle className="w-4 h-4" /> {item.error}
        </div>
      ) : (
        <ResultBlock result={item.result} />
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
let _id = 0
const EXAMPLES = [
  'Who entered the building between 10am and 11am?',
  'Find the person in a red jacket near the entrance.',
  'Show clips of someone appearing in multiple cameras.',
  'Who was at the staircase around 3pm?',
]

export default function QueryPage() {
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [threadId] = useState(() => `thread_${Date.now()}`)
  const inputRef = useRef()
  const bottomRef = useRef()

  const submit = useCallback(async () => {
    const q = input.trim()
    if (!q) return
    setInput('')
    const id = ++_id
    setHistory(h => [...h, { id, query: q, loading: true, result: null, error: null }])
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    try {
      const res = await fetch(`${API}/api/v1/agent/query`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, thread_id: threadId, user_id: 'ui_user', include_rows: false, include_node_trace: true, top_k_rows: 8, response_style: 'natural' }),
      })
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText) }
      const data = await res.json()
      setHistory(h => h.map(it => it.id === id ? { ...it, loading: false, result: data } : it))
    } catch (err) {
      setHistory(h => h.map(it => it.id === id ? { ...it, loading: false, error: err.message } : it))
    }
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
  }, [input, threadId])

  const onKey = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }

  return (
    <div className="flex flex-col h-full bg-slate-50">
      <DatabaseSelector />
      {/* Conversation */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5 max-w-4xl w-full mx-auto">
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 pb-16">
            <div className="text-center">
              <div className="w-14 h-14 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-3">
                <Search className="w-7 h-7 text-blue-500" />
              </div>
              <h2 className="text-slate-700 font-semibold text-base mb-1">Surveillance Query Agent</h2>
              <p className="text-slate-400 text-sm">Ask a natural-language question about the ingested videos.</p>
            </div>
            <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
              {EXAMPLES.map(ex => (
                <button key={ex} onClick={() => { setInput(ex); inputRef.current?.focus() }}
                  className="text-left px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-xs text-slate-500
                             hover:text-slate-700 hover:border-blue-300 hover:bg-blue-50 transition-all shadow-sm">
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : history.map(item => <HistoryItem key={item.id} item={item} />)}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="flex-shrink-0 border-t border-slate-200 bg-white px-6 py-4 shadow-sm">
        <div className="max-w-4xl mx-auto flex gap-3 items-end">
          <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={onKey}
            rows={1} placeholder="Ask about people, locations, times, appearances…"
            className="flex-1 border border-slate-300 rounded-xl px-4 py-3 text-sm text-slate-700
                       placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none bg-white"
            style={{ minHeight: '44px', maxHeight: '160px' }}
            onInput={e => { e.target.style.height = 'auto'; e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px` }}
          />
          <button onClick={submit} disabled={!input.trim()}
            className="flex-shrink-0 p-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm">
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-center text-xs text-slate-400 mt-2">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}

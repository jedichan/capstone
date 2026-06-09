import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Activity } from 'lucide-react'
import UploadPage from './pages/UploadPage'
import QueryPage from './pages/QueryPage'

function TopNav() {
  const link = ({ isActive }) =>
    `px-3 py-1.5 rounded text-sm transition-colors ${
      isActive
        ? 'text-white bg-white/15 font-medium'
        : 'text-slate-300 hover:text-white hover:bg-white/10'
    }`

  return (
    <nav className="bg-[#1a2744] h-13 flex items-center px-6 gap-6 shadow-md flex-shrink-0" style={{height:'52px'}}>
      <div className="flex items-center gap-2 mr-2">
        <div className="w-7 h-7 bg-blue-500 rounded-md flex items-center justify-center">
          <Activity className="w-4 h-4 text-white" />
        </div>
        <span className="text-white font-bold text-sm tracking-wide">SurveillanceIQ</span>
      </div>

      <NavLink to="/" end className={link}>Video Ingestion</NavLink>
      <NavLink to="/query" className={link}>Query Agent</NavLink>

      <div className="ml-auto flex items-center gap-2">
        <span className="bg-green-500 text-white text-xs font-semibold px-2.5 py-0.5 rounded-full">
          ● Pipeline Ready
        </span>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopNav />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/query" element={<QueryPage />} />
        </Routes>
      </main>
    </div>
  )
}

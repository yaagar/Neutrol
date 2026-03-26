// App.jsx — root component
// Renders the two-column shell: a fixed left sidebar and a scrollable main content area.
// React Router's BrowserRouter wraps everything so child components can use <Link> and
// <Route> to navigate between pages without a full page reload.
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import StatusBar from './components/StatusBar'
import ImportPage from './pages/ImportPage'
import PreprocessPage from './pages/PreprocessPage'
import LabelPage from './pages/LabelPage'
import TrainPage from './pages/TrainPage'
import InferencePage from './pages/InferencePage'
import ExportPage from './pages/ExportPage'

export default function App() {
  return (
    // BrowserRouter gives every child access to routing context (useNavigate, Link, etc.)
    <BrowserRouter>
      {/* Full-viewport flex row: sidebar on left, content column on right */}
      <div className="flex h-screen bg-gray-950 text-gray-100">

        {/* Left sidebar — fixed width, never scrolls */}
        <Sidebar />

        {/* Right side: StatusBar pinned at top, scrollable page content below */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* StatusBar always visible regardless of which page is active */}
          <StatusBar />

          {/* Page content — scrolls independently of the sidebar and status bar */}
          <main className="flex-1 overflow-y-auto p-8">
            <Routes>
              <Route path="/" element={<ImportPage />} />
              <Route path="/import" element={<ImportPage />} />
              <Route path="/preprocess" element={<PreprocessPage />} />
              <Route path="/label" element={<LabelPage />} />
              <Route path="/train" element={<TrainPage />} />
              <Route path="/inference" element={<InferencePage />} />
              <Route path="/export" element={<ExportPage />} />
            </Routes>
          </main>

        </div>

      </div>
    </BrowserRouter>
  )
}

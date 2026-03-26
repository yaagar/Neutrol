// Sidebar.jsx — left navigation panel
// Renders the app name and one nav link per workflow step.
// NavLink from react-router-dom automatically adds an "active" class when the
// current URL matches the link's `to` path — we use that to highlight the current page.
import { NavLink } from 'react-router-dom'

// The full ordered workflow — Import first, Export last.
// Each entry maps a display label to a URL path.
const NAV_ITEMS = [
  { label: 'Import',     path: '/import' },
  { label: 'Preprocess', path: '/preprocess' },
  { label: 'Label',      path: '/label' },
  { label: 'Train',      path: '/train' },
  { label: 'Inference',  path: '/inference' },
  { label: 'Export',     path: '/export' },
]

export default function Sidebar() {
  return (
    // w-56 = 224px fixed sidebar, full height, dark background to visually separate from content
    <nav className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col py-6 px-4 shrink-0">

      {/* App title */}
      <div className="mb-8">
        <h1 className="text-xl font-bold text-purple-400">OpenEEG</h1>
        <p className="text-xs text-gray-500 mt-1">EEG Development Platform</p>
      </div>

      {/* Navigation links — one per page */}
      <ul className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ label, path }) => (
          <li key={path}>
            {/* NavLink's className prop receives { isActive } — we use it to style the active link */}
            <NavLink
              to={path}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-purple-600 text-white'           // active page: purple highlight
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'  // inactive: subtle hover
                }`
              }
            >
              {label}
            </NavLink>
          </li>
        ))}
      </ul>

    </nav>
  )
}

// useWebSocket.js — manages a WebSocket connection to the backend
//
// Why a custom hook for WebSocket?
//   - WebSocket connections need to be opened once and cleaned up when the component
//     unmounts. React's useEffect with a cleanup function is the right tool.
//   - Centralizing this logic means InferencePage and TrainPage both use the same
//     connection management without duplicating the open/close/error logic.
//
// Stub for Session 1 — the actual WebSocket logic is implemented in Session 4 (training)
// and Session 6 (inference streaming).
import { useEffect, useRef, useState } from 'react'

// url     — the WebSocket URL to connect to, e.g. "ws://localhost:8000/ws/inference/stream"
// onMessage — callback called with each parsed JSON message from the server
export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null)               // holds the WebSocket instance across renders
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Don't connect if no URL provided (e.g., before a session is loaded)
    if (!url) return

    // Open the WebSocket connection
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)

    // Parse incoming JSON and forward to the caller's handler
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        onMessage(msg)
      } catch {
        // Non-JSON message — ignore in MVP, log in production
      }
    }

    ws.onerror = () => setError('WebSocket connection error')
    ws.onclose = () => setConnected(false)

    // Cleanup: close the socket when the component using this hook unmounts.
    // Without this, the socket stays open in the background — wasting backend resources.
    return () => ws.close()
  }, [url]) // Re-run only if the URL changes (i.e., connecting to a different endpoint)

  // send() — lets the component push a message to the server
  const send = (data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }

  return { connected, error, send }
}

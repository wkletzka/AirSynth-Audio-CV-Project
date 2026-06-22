// import { useState } from 'react'
import './App.css'

function App() {
  // const [count, setCount] = useState(0)

  return (
    <div className="synth-container">
      <h1>Air Piano Dashboard</h1>
      
      {/* This image tag acts as your live video player! */}
      <div className="camera-feed">
        <img 
            src="http://localhost:8000/video_feed" 
            alt="RealSense Camera Stream" 
            width="640" 
            height="480"
        />
      </div>

    </div>
  )
}

export default App

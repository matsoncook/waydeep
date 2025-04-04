import { useEffect, useRef, useState } from 'react';

type Message = {
  sender: 'user' | 'bot';
  text: string;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
  
    ws.current.onmessage = (event) => {
      setMessages((prev) => {
        const lastIndex = prev.length - 1;
        const last = prev[lastIndex];
  
        if (last?.sender === 'bot') {
          const updated = [...prev];
          updated[lastIndex] = { ...last, text: last.text + event.data };
          return updated;
        } else {
          return [...prev, { sender: 'bot', text: event.data }];
        }
      });
    };
  
    return () => ws.current?.close();
  }, []); // ✅ Only run once on component mount

  const send = () => {
    if (!input.trim() || !ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    setMessages([...messages, { sender: 'user', text: input }]);
    ws.current.send(input);
    setInput('');
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <div style={{ marginBottom: '1rem' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
            <strong>{msg.sender === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && send()}
        placeholder="Type a message..."
        style={{ width: '80%' }}
      />
      <button onClick={send} disabled={!isConnected} style={{ marginLeft: '1rem' }}>
        Send
      </button>
    </div>
  );
}

export default App;

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
    ws.current = new WebSocket('ws://ocsswe-ai-dev:8000/ws');
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
  
    ws.current.onmessage = (event) => {
      setMessages((prev) => {
        const lastIndex = prev.length - 1;
        const last = prev[lastIndex];
  
        if (last?.sender === 'bot') {

          const updated = [...prev];
          updated[lastIndex] = { ...last, text: last.text + event.data.slice(0, -1) };
          return updated;
        } else {
          return [...prev, { sender: 'bot', text: '\n'+ event.data.slice(0, -1) }];
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
    <div 
    id="OuterDiv"
   >
      <div style={{ marginBottom: '1rem', flex: 1 }}>
        {messages.map((msg, idx) => (
          
          <div key={idx} 
          className={msg.sender === 'bot' ? 'bot-output' : 'user-input'}
          style={{ textAlign: msg.sender === 'user' ? 'right' : 'left',whiteSpace: 'pre-wrap' }} >
            < strong style={{ fontSize: '1.5rem' }}>{msg.sender === 'user' ? '🛫' : '🐋'}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <div id="InputArea">
        <textarea
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
    </div>
  );
}

export default App;

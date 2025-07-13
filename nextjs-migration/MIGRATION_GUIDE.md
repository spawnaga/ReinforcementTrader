# AI Trading System Migration Guide
## Flask + JavaScript → Next.js + TypeScript + FastAPI

This guide demonstrates how to migrate your trading system to a modern tech stack.

## Migration Overview

### Backend: Flask → FastAPI
- **REST API**: All Flask routes converted to FastAPI endpoints with automatic validation
- **WebSockets**: Flask-SocketIO → Native FastAPI WebSockets
- **Type Safety**: Pydantic models for request/response validation
- **Performance**: Async/await support for better concurrency
- **Documentation**: Automatic OpenAPI/Swagger docs

### Frontend: JavaScript → TypeScript + Next.js
- **Type Safety**: Full TypeScript coverage with interfaces
- **Component Architecture**: React components with hooks
- **State Management**: Zustand for global state (replaces manual DOM updates)
- **Real-time Updates**: Custom WebSocket hook
- **Modern Styling**: Tailwind CSS for responsive design

## Key Files Created

### Backend (FastAPI)
```
nextjs-migration/backend/
├── main.py           # FastAPI application with WebSocket support
├── models.py         # Pydantic models for type validation
```

### Frontend (Next.js)
```
nextjs-migration/frontend/
├── package.json                              # Dependencies
├── tsconfig.json                            # TypeScript config
├── src/
│   ├── types/index.ts                       # TypeScript interfaces
│   ├── store/tradingStore.ts                # Zustand state management
│   ├── hooks/useWebSocket.ts                # WebSocket custom hook
│   └── components/Dashboard/
│       └── TradingDashboard.tsx             # Main dashboard component
```

## Migration Steps

### Phase 1: Backend API Migration (Week 1-2)
1. **Set up FastAPI project**
   - Install dependencies: `pip install fastapi uvicorn sqlalchemy psycopg2-binary`
   - Create main.py with basic structure

2. **Migrate database models**
   - Keep SQLAlchemy models mostly unchanged
   - Add Pydantic models for API validation

3. **Convert routes**
   - Map Flask routes to FastAPI endpoints
   - Add proper type hints and validation

4. **Implement WebSockets**
   - Replace Flask-SocketIO with FastAPI WebSockets
   - Update connection management

### Phase 2: Frontend Migration (Week 3-4)
1. **Initialize Next.js project**
   ```bash
   npx create-next-app@latest frontend --typescript --tailwind --app
   cd frontend
   npm install socket.io-client zustand chart.js react-chartjs-2 three @react-three/fiber
   ```

2. **Create TypeScript types**
   - Define interfaces for all data models
   - Add type definitions for API responses

3. **Build React components**
   - Convert HTML templates to React components
   - Implement hooks for state management

4. **Set up state management**
   - Create Zustand stores for global state
   - Replace DOM manipulation with React state

### Phase 3: Feature Parity (Week 5)
1. **Charts migration**
   - Use react-chartjs-2 for Chart.js integration
   - Implement real-time chart updates

2. **3D visualization**
   - Use @react-three/fiber for Three.js in React
   - Convert vanilla Three.js to React Three Fiber

3. **Neural network visualization**
   - Create React component with Canvas API
   - Implement animation loops with useEffect

### Phase 4: Integration & Testing (Week 6)
1. **API integration**
   - Update API endpoints to FastAPI URLs
   - Test all CRUD operations

2. **WebSocket testing**
   - Verify real-time updates
   - Test session management

3. **Performance optimization**
   - Implement code splitting
   - Add lazy loading for heavy components

## Code Examples

### FastAPI Endpoint (Before: Flask)
```python
# Flask
@app.route('/api/training/start', methods=['POST'])
def start_training():
    data = request.json
    # ... validation logic
    return jsonify(result)

# FastAPI
@app.post('/api/training/start')
async def start_training(request: TrainingStartRequest):
    # Automatic validation!
    return {"session_id": "...", "status": "started"}
```

### React Component (Before: Vanilla JS)
```javascript
// Before: Vanilla JavaScript
document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/api/training/start', {...})
});

// After: React + TypeScript
const handleStartTraining = async () => {
    const response = await fetch('http://localhost:8000/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            algorithm_type: selectedAlgorithm,
            total_episodes: totalEpisodes,
        }),
    });
};
```

### State Management (Before: Global Variables)
```javascript
// Before: Global variables
let isTraining = false;
let currentSession = null;

// After: Zustand store
const useTradingStore = create((set) => ({
    isTraining: false,
    activeSession: null,
    setIsTraining: (value) => set({ isTraining: value }),
}));
```

## Benefits of Migration

1. **Type Safety**: Catch errors at compile time
2. **Better Performance**: Next.js SSR/SSG, FastAPI async
3. **Developer Experience**: Better tooling, auto-completion
4. **Maintainability**: Clear component structure
5. **Scalability**: Microservices-ready architecture
6. **Modern Stack**: Easier to hire developers

## Running the New System

### Backend
```bash
cd nextjs-migration/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd nextjs-migration/frontend
npm install
npm run dev
```

## Next Steps

1. **Complete component migration**: Convert all JavaScript files to TypeScript components
2. **API integration**: Update all endpoints to use FastAPI
3. **Testing**: Add unit and integration tests
4. **Deployment**: Configure for production deployment
5. **Documentation**: Update API documentation

The migration provides a solid foundation for future enhancements while maintaining all current functionality.
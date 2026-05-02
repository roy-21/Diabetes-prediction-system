import { useState } from 'react'

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50,
    model_type: 'ensemble'
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'model_type' ? value : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12 flex flex-col items-center justify-center min-h-screen">
      
      <div className="text-center mb-10 animate-float">
        <h1 className="text-5xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
          AI Diabetes Predictor
        </h1>
        <p className="text-gray-300 text-lg">Advanced ensemble and deep learning models for accurate diagnosis</p>
      </div>

      <div className="w-full max-w-4xl grid md:grid-cols-2 gap-8">
        {/* Form Panel */}
        <div className="glass-panel p-8">
          <h2 className="text-2xl font-bold mb-6 border-b border-white/20 pb-2">Patient Details</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Pregnancies</label>
                <input type="number" name="Pregnancies" value={formData.Pregnancies} onChange={handleChange} className="glass-input" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Glucose</label>
                <input type="number" name="Glucose" value={formData.Glucose} onChange={handleChange} className="glass-input" step="0.1" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Blood Pressure</label>
                <input type="number" name="BloodPressure" value={formData.BloodPressure} onChange={handleChange} className="glass-input" step="0.1" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Skin Thickness</label>
                <input type="number" name="SkinThickness" value={formData.SkinThickness} onChange={handleChange} className="glass-input" step="0.1" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Insulin</label>
                <input type="number" name="Insulin" value={formData.Insulin} onChange={handleChange} className="glass-input" step="0.1" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">BMI</label>
                <input type="number" name="BMI" value={formData.BMI} onChange={handleChange} className="glass-input" step="0.1" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Pedigree Function</label>
                <input type="number" name="DiabetesPedigreeFunction" value={formData.DiabetesPedigreeFunction} onChange={handleChange} className="glass-input" step="0.001" required />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Age</label>
                <input type="number" name="Age" value={formData.Age} onChange={handleChange} className="glass-input" required />
              </div>
            </div>

            <div className="mt-6">
              <label className="block text-sm text-gray-300 mb-1">AI Model Strategy</label>
              <select name="model_type" value={formData.model_type} onChange={handleChange} className="glass-input [&>option]:bg-gray-800">
                <option value="ensemble">Gradient Boosting Ensemble (XGBoost/LGBM)</option>
                <option value="tabnet">Deep Learning (TabNet)</option>
              </select>
            </div>

            <button type="submit" disabled={loading} className="glass-button mt-6">
              {loading ? 'Analyzing...' : 'Predict Diabetes Risk'}
            </button>
            {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
          </form>
        </div>

        {/* Results Panel */}
        <div className="glass-panel p-8 flex flex-col justify-center items-center text-center">
          {result ? (
            <div className="animate-fade-in">
              <div className={`w-32 h-32 rounded-full flex items-center justify-center mx-auto mb-6 text-4xl shadow-[0_0_40px_rgba(0,0,0,0.5)] ${result.prediction === 1 ? 'bg-red-500/20 shadow-red-500/50 text-red-400 border-2 border-red-500/50' : 'bg-green-500/20 shadow-green-500/50 text-green-400 border-2 border-green-500/50'}`}>
                {result.prediction === 1 ? '⚠️' : '✅'}
              </div>
              
              <h2 className="text-3xl font-bold mb-2">{result.message}</h2>
              <p className="text-xl text-gray-300 mb-6">
                Probability: <span className="font-bold text-white">{(result.probability * 100).toFixed(1)}%</span>
              </p>
              
              <div className="bg-black/20 rounded-xl p-4 w-full text-left">
                <h3 className="text-sm uppercase tracking-wider text-gray-400 mb-3 font-semibold">Key Factors (SHAP)</h3>
                <ul className="space-y-2">
                  {result.top_contributors.map((contrib, idx) => (
                    <li key={idx} className="flex justify-between items-center text-sm border-b border-white/5 pb-2 last:border-0">
                      <span>{contrib.Feature}</span>
                      <span className={contrib.SHAP_Value > 0 ? 'text-red-400' : 'text-green-400'}>
                        {contrib.SHAP_Value > 0 ? '↑' : '↓'} {Math.abs(contrib.SHAP_Value).toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-4 text-xs text-gray-400">
                Processed via: {result.model_used.toUpperCase()}
              </div>
            </div>
          ) : (
            <div className="text-gray-400 flex flex-col items-center">
              <svg className="w-16 h-16 mb-4 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
              <p>Enter patient details and run the prediction to see results and SHAP insights here.</p>
            </div>
          )}
        </div>
      </div>
      
    </div>
  )
}

export default App

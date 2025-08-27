import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function ForecastChart({ history = [], forecast = [] }) {
  const hist = history.map(d => ({ ...d, type: "History" }));
  const fcst = forecast.map(d => ({ ...d, type: "Forecast" }));
  const data = [...hist, ...fcst];

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="total" stroke="#8884d8" name="History" dot={false}
              connectNulls={true} isAnimationActive={false}
              strokeDasharray="0" data={hist} />
        <Line type="monotone" dataKey="total" stroke="#82ca9d" name="Forecast" dot={true}
              connectNulls={true} isAnimationActive={false}
              strokeDasharray="5 5" data={fcst} />
      </LineChart>
    </ResponsiveContainer>
  );
}

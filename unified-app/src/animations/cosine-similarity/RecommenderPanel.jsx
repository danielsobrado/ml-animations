import React, { useState } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer } from 'recharts';

export default function RecommenderPanel() {
    // User preferences (0-10 scale)
    const [userA, setUserA] = useState({ action: 8, comedy: 3, romance: 2, scifi: 7, drama: 4 });
    const [userB] = useState({ action: 7, comedy: 4, romance: 3, scifi: 8, drama: 3 }); // Similar
    const [userC] = useState({ action: 2, comedy: 9, romance: 8, scifi: 1, drama: 6 }); // Different

    const calculateSimilarity = (u1, u2) => {
        const dotProduct = u1.action * u2.action + u1.comedy * u2.comedy + u1.romance * u2.romance + u1.scifi * u2.scifi + u1.drama * u2.drama;
        const mag1 = Math.sqrt(u1.action ** 2 + u1.comedy ** 2 + u1.romance ** 2 + u1.scifi ** 2 + u1.drama ** 2);
        const mag2 = Math.sqrt(u2.action ** 2 + u2.comedy ** 2 + u2.romance ** 2 + u2.scifi ** 2 + u2.drama ** 2);
        return dotProduct / (mag1 * mag2);
    };

    const simB = calculateSimilarity(userA, userB);
    const simC = calculateSimilarity(userA, userC);

    // Prepare data for radar chart
    const radarData = [
        { genre: 'Action', A: userA.action, B: userB.action, C: userC.action },
        { genre: 'Comedy', A: userA.comedy, B: userB.comedy, C: userC.comedy },
        { genre: 'Romance', A: userA.romance, B: userB.romance, C: userC.romance },
        { genre: 'Sci-Fi', A: userA.scifi, B: userB.scifi, C: userC.scifi },
        { genre: 'Drama', A: userA.drama, B: userB.drama, C: userC.drama }
    ];

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent mb-4">
                    Movie Matcher
                </h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Netflix doesn't guess. It calculates similarity between your taste and millions of users.
                    <br />
                    Who are you most compatible with?
                </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8 w-full max-w-6xl">
                {/* Radar Chart */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="text-center font-bold text-slate-300 mb-4">Genre Preference Profiles</h3>
                    <ResponsiveContainer width="100%" height={400}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#475569" />
                            <PolarAngleAxis dataKey="genre" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                            <PolarRadiusAxis domain={[0, 10]} tick={{ fill: '#94a3b8' }} />
                            <Radar name="You (A)" dataKey="A" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} strokeWidth={2} />
                            <Radar name="User B" dataKey="B" stroke="#a855f7" fill="#a855f7" fillOpacity={0.2} strokeWidth={2} />
                            <Radar name="User C" dataKey="C" stroke="#ec4899" fill="#ec4899" fillOpacity={0.2} strokeWidth={2} />
                            <Legend wrapperStyle={{ color: '#e2e8f0' }} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                {/* Controls & Results */}
                <div className="flex flex-col gap-6">
                    {/* Your Preferences */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-cyan-500/50">
                        <h3 className="font-bold text-cyan-400 mb-4 uppercase text-sm">Your Movie Taste (User A)</h3>
                        <div className="space-y-3">
                            {Object.entries(userA).map(([genre, value]) => (
                                <div key={genre}>
                                    <label className="flex justify-between text-sm mb-1 capitalize">
                                        {genre}: <span className="text-cyan-400 font-mono">{value}</span>
                                    </label>
                                    <input
                                        type="range" min="0" max="10" step="1"
                                        value={value}
                                        onChange={(e) => setUserA({ ...userA, [genre]: Number(e.target.value) })}
                                        className="w-full accent-cyan-400"
                                    />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Similarity Scores */}
                    <div className="space-y-4">
                        <div className={`p-6 rounded-xl border-2 transition-all ${simB > simC ? 'bg-gradient-to-br from-purple-900/50 to-purple-800/30 border-purple-500' : 'bg-slate-800/50 border-slate-700'}`}>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="font-bold text-purple-300">Match with User B</h4>
                                {simB > 0.9 && <span className="text-xs bg-purple-500 px-2 py-1 rounded-full font-bold">BEST MATCH!</span>}
                            </div>
                            <div className="text-4xl font-mono font-bold text-purple-200">
                                {(simB * 100).toFixed(1)}%
                            </div>
                            <p className="text-xs text-slate-400 mt-2">Cosine Similarity: {simB.toFixed(3)}</p>
                        </div>

                        <div className={`p-6 rounded-xl border-2 transition-all ${simC > simB ? 'bg-gradient-to-br from-pink-900/50 to-pink-800/30 border-pink-500' : 'bg-slate-800/50 border-slate-700'}`}>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="font-bold text-pink-300">Match with User C</h4>
                                {simC > 0.9 && <span className="text-xs bg-pink-500 px-2 py-1 rounded-full font-bold">BEST MATCH!</span>}
                            </div>
                            <div className="text-4xl font-mono font-bold text-pink-200">
                                {(simC * 100).toFixed(1)}%
                            </div>
                            <p className="text-xs text-slate-400 mt-2">Cosine Similarity: {simC.toFixed(3)}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

import React, { useState } from 'react';
import { motion } from 'framer-motion';

// Small array to force collisions easily
const SMALL_SIZE = 8;

const hash1 = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
    return Math.abs(hash);
};

const hash2 = (str) => {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) hash = (hash * 33) ^ str.charCodeAt(i);
    return Math.abs(hash);
};

export default function CollisionPanel() {
    const [bits, setBits] = useState(Array(SMALL_SIZE).fill(0));
    const [addedWords, setAddedWords] = useState([]);
    const [checkWord, setCheckWord] = useState('');
    const [message, setMessage] = useState(null);

    const getIndices = (word) => [hash1(word) % SMALL_SIZE, hash2(word) % SMALL_SIZE];

    const addWord = (word) => {
        const indices = getIndices(word);
        const newBits = [...bits];
        indices.forEach(idx => newBits[idx] = 1);
        setBits(newBits);
        setAddedWords([...addedWords, { word, indices }]);
    };

    const checkCollision = () => {
        if (!checkWord) return;
        const indices = getIndices(checkWord);
        const allSet = indices.every(idx => bits[idx] === 1);
        const actuallyAdded = addedWords.some(w => w.word === checkWord);

        if (allSet && !actuallyAdded) {
            setMessage({ type: 'collision', text: `⚠️ FALSE POSITIVE! "${checkWord}" looks added (bits ${indices.join(', ')} are 1), but it's just a collision from other words!` });
        } else if (allSet && actuallyAdded) {
            setMessage({ type: 'info', text: `"${checkWord}" is in the set.` });
        } else {
            setMessage({ type: 'info', text: `"${checkWord}" is definitely not in the set.` });
        }
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">False Positive Lab</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    This filter is tiny (size 8). Add a few words to fill it up, then try to find a "False Positive" (a word you didn't add, but the filter says "Probably Yes").
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <button onClick={() => addWord("Cat")} className="px-4 py-2 bg-blue-100 text-blue-800 rounded font-bold hover:bg-blue-200">Add "Cat"</button>
                <button onClick={() => addWord("Dog")} className="px-4 py-2 bg-blue-100 text-blue-800 rounded font-bold hover:bg-blue-200">Add "Dog"</button>
                <button onClick={() => addWord("Fish")} className="px-4 py-2 bg-blue-100 text-blue-800 rounded font-bold hover:bg-blue-200">Add "Fish"</button>
                <button onClick={() => { setBits(Array(SMALL_SIZE).fill(0)); setAddedWords([]); setMessage(null); }} className="px-4 py-2 bg-slate-200 text-slate-700 rounded font-bold hover:bg-slate-300">Reset</button>
            </div>

            {/* Bit Array */}
            <div className="flex gap-2 mb-8">
                {bits.map((bit, i) => (
                    <div key={i} className={`w-12 h-12 rounded border-2 flex items-center justify-center font-bold text-xl transition-colors ${bit === 1 ? 'bg-indigo-600 text-white border-indigo-700' : 'bg-slate-100 text-slate-700 dark:text-slate-300 border-slate-200'}`}>
                        {bit}
                    </div>
                ))}
            </div>

            {/* Check Area */}
            <div className="flex gap-4 mb-4">
                <input
                    type="text"
                    value={checkWord}
                    onChange={(e) => setCheckWord(e.target.value)}
                    placeholder="Try 'Bird' or 'Cow'..."
                    className="px-4 py-2 border-2 border-slate-300 rounded font-bold"
                />
                <button onClick={checkCollision} className="px-6 py-2 bg-teal-600 text-white rounded font-bold hover:bg-teal-700">Check</button>
            </div>

            {message && (
                <div className={`p-4 rounded-lg font-bold text-center max-w-xl ${message.type === 'collision' ? 'bg-red-100 text-red-900 border-2 border-red-200' : 'bg-slate-100 text-slate-700'}`}>
                    {message.text}
                </div>
            )}

            <div className="mt-8 text-sm text-slate-700 dark:text-slate-500">
                <p>Added Words: {addedWords.map(w => w.word).join(', ') || '(None)'}</p>
            </div>
        </div>
    );
}

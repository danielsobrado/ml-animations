import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function ConditionalPanel() {
    const [selectedCard, setSelectedCard] = useState(null);
    const [filterFaceCards, setFilterFaceCards] = useState(false);

    // Card deck
    const suits = ['♠', '♥', '♦', '♣'];
    const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
    const faceCards = ['J', 'Q', 'K'];
    const kings = ['K'];

    const allCards = suits.flatMap(suit =>
        ranks.map(rank => ({ suit, rank, isFace: faceCards.includes(rank), isKing: kings.includes(rank) }))
    );

    const visibleCards = filterFaceCards ? allCards.filter(c => c.isFace) : allCards;
    const kingCount = visibleCards.filter(c => c.isKing).length;
    const totalCount = visibleCards.length;

    const probability = totalCount > 0 ? (kingCount / totalCount) : 0;

    const getCardColor = (suit) => {
        return suit === '♥' || suit === '♦' ? 'text-red-500' : 'text-slate-900';
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-400 mb-4">Conditional Probability</h2>
                <p className="text-lg text-slate-300 leading-relaxed mb-4">
                    <strong>P(A|B)</strong> = "Probability of A, <em>given</em> B has occurred"
                </p>
                <div className="bg-slate-800 p-4 rounded-lg font-mono text-sm">
                    <p className="text-emerald-300">P(A|B) = P(A ∩ B) / P(B)</p>
                    <p className="text-slate-400 mt-2 text-xs">
                        Conditioning on B "shrinks" the sample space to only outcomes where B is true.
                    </p>
                </div>
            </div>

            {/* Example */}
            <div className="bg-slate-800 p-6 rounded-xl border border-emerald-500/50 w-full max-w-4xl mb-8">
                <h3 className="font-bold text-white mb-4 text-center text-xl">
                    Example: Drawing from a Deck
                </h3>
                <p className="text-slate-300 text-center mb-6">
                    What's the probability of drawing a <strong className="text-yellow-400">King</strong>
                    {filterFaceCards && <span>, given it's a <strong className="text-cyan-400">Face Card</strong>?</span>}
                    {!filterFaceCards && <span>?</span>}
                </p>

                {/* Toggle */}
                <div className="flex items-center justify-center gap-4 mb-6">
                    <span className="text-slate-400">Show all cards</span>
                    <button
                        onClick={() => setFilterFaceCards(!filterFaceCards)}
                        className={`relative w-20 h-10 rounded-full transition-colors ${filterFaceCards ? 'bg-cyan-500' : 'bg-slate-600'
                            }`}
                    >
                        <motion.div
                            className="absolute top-1 left-1 w-8 h-8 bg-white rounded-full shadow-lg"
                            animate={{ x: filterFaceCards ? 40 : 0 }}
                            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                    </button>
                    <span className="text-cyan-400 font-bold">Filter: Face Cards only</span>
                </div>

                {/* Cards Grid */}
                <div className="grid grid-cols-13 gap-1 mb-6 max-h-[300px] overflow-y-auto">
                    {visibleCards.map((card, idx) => (
                        <motion.div
                            key={idx}
                            layout
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            className={`aspect-[2/3] bg-white rounded border-2 flex items-center justify-center text-2xl font-bold cursor-pointer transition-all ${card.isKing ? 'border-yellow-400 shadow-lg shadow-yellow-400/50' : 'border-slate-300'
                                }`}
                            onClick={() => setSelectedCard(card)}
                        >
                            <span className={getCardColor(card.suit)}>
                                {card.rank}{card.suit}
                            </span>
                        </motion.div>
                    ))}
                </div>

                {/* Calculation */}
                <div className="bg-slate-900 p-6 rounded-lg border border-slate-700">
                    <div className="grid md:grid-cols-3 gap-4 text-center">
                        <div>
                            <div className="text-slate-400 text-sm mb-2">Sample Space</div>
                            <div className="text-3xl font-bold text-white">{totalCount}</div>
                            <div className="text-xs text-slate-500 mt-1">
                                {filterFaceCards ? 'Face cards' : 'All cards'}
                            </div>
                        </div>
                        <div>
                            <div className="text-slate-400 text-sm mb-2">Kings</div>
                            <div className="text-3xl font-bold text-yellow-400">{kingCount}</div>
                            <div className="text-xs text-slate-500 mt-1">
                                {filterFaceCards ? 'Kings among faces' : 'Kings in deck'}
                            </div>
                        </div>
                        <div>
                            <div className="text-slate-400 text-sm mb-2">
                                {filterFaceCards ? 'P(King | Face)' : 'P(King)'}
                            </div>
                            <div className="text-3xl font-bold text-emerald-400">
                                {(probability * 100).toFixed(1)}%
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                {kingCount}/{totalCount}
                            </div>
                        </div>
                    </div>

                    {filterFaceCards && (
                        <div className="mt-6 p-4 bg-emerald-900/30 rounded-lg border border-emerald-700">
                            <p className="text-emerald-300 text-sm text-center">
                                ✅ Conditioning on "Face Card" reduced the sample space from 52 → 12 cards.
                                <br />
                                The probability of King increased from 7.7% → 25%!
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

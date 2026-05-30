import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function ConditionalPanel() {
    const [selectedCard, setSelectedCard] = useState(null);
    const [filterFaceCards, setFilterFaceCards] = useState(false);

    const suits = ['\u2660', '\u2665', '\u2666', '\u2663'];
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
    const deckColumns = filterFaceCards ? 'repeat(3, minmax(3rem, 4.25rem))' : 'repeat(13, minmax(2.35rem, 3.6rem))';

    const getCardColor = (suit) => {
        return suit === '\u2665' || suit === '\u2666' ? 'text-red-600' : 'text-slate-900';
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-400 mb-4">Conditional Probability</h2>
                <p className="text-lg text-slate-300 leading-relaxed mb-4">
                    <strong>P(A|B)</strong> = "Probability of A, <em>given</em> B has occurred"
                </p>
                <div className="bg-slate-800 p-4 rounded-lg font-mono text-sm">
                    <p className="text-emerald-300">P(A|B) = P(A &cap; B) / P(B)</p>
                    <p className="text-slate-400 mt-2 text-xs">
                        Conditioning on B "shrinks" the sample space to only outcomes where B is true.
                    </p>
                </div>
            </div>

            <div className="bg-slate-800 p-6 rounded-xl border border-emerald-500/50 w-full max-w-4xl mb-8">
                <h3 className="font-bold text-white mb-4 text-center text-xl">
                    Example: Drawing from a Deck
                </h3>
                <p className="text-slate-300 text-center mb-6">
                    What's the probability of drawing a <strong className="text-yellow-400">King</strong>
                    {filterFaceCards && <span>, given it's a <strong className="text-cyan-400">Face Card</strong>?</span>}
                    {!filterFaceCards && <span>?</span>}
                </p>

                <div className="flex items-center justify-center gap-4 mb-6">
                    <span className="text-slate-400">Show all cards</span>
                    <button
                        type="button"
                        onClick={() => setFilterFaceCards(!filterFaceCards)}
                        className={`relative w-20 h-10 rounded-full transition-colors ${filterFaceCards ? 'bg-cyan-500' : 'bg-slate-600'
                            }`}
                        aria-pressed={filterFaceCards}
                    >
                        <motion.div
                            className="absolute top-1 left-1 w-8 h-8 bg-white rounded-full shadow-lg"
                            animate={{ x: filterFaceCards ? 40 : 0 }}
                            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                    </button>
                    <span className="text-cyan-400 font-bold">Filter: Face Cards only</span>
                </div>

                <div className="mb-6 rounded-lg border border-slate-700 bg-slate-950/40 p-4">
                    <div
                        className="grid justify-center gap-2 overflow-x-auto pb-1"
                        style={{ gridTemplateColumns: deckColumns }}
                    >
                        {visibleCards.map((card, idx) => {
                            const isSelected = selectedCard?.rank === card.rank && selectedCard?.suit === card.suit;

                            return (
                                <motion.button
                                    key={`${card.rank}-${card.suit}`}
                                    type="button"
                                    layout
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 8 }}
                                    transition={{ delay: idx * 0.01 }}
                                    className={`aspect-[2/3] min-h-[4.25rem] rounded-md border-2 bg-white px-1 py-1 text-left shadow-sm transition-all ${card.isKing ? 'border-yellow-500 ring-2 ring-yellow-300/40' : 'border-slate-300'
                                        } ${isSelected ? 'scale-105 ring-2 ring-cyan-400' : 'hover:-translate-y-1 hover:shadow-md'}`}
                                    onClick={() => setSelectedCard(card)}
                                    aria-label={`${card.rank} ${card.suit}`}
                                >
                                    <span className={`block text-base font-bold leading-none ${getCardColor(card.suit)}`}>
                                        {card.rank}
                                    </span>
                                    <span className={`mt-1 flex h-full items-center justify-center text-2xl font-bold ${getCardColor(card.suit)}`}>
                                        {card.suit}
                                    </span>
                                </motion.button>
                            );
                        })}
                    </div>
                    <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400">
                        <span><strong className="text-yellow-400">Gold outline</strong> = event A: King</span>
                        <span><strong className="text-cyan-400">{filterFaceCards ? 'Current grid' : 'Toggle on'}</strong> = condition B: Face Card</span>
                        {selectedCard && <span>Selected: <strong>{selectedCard.rank}{selectedCard.suit}</strong></span>}
                    </div>
                </div>

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
                                Conditioning on "Face Card" reduced the sample space from 52 to 12 cards.
                                <br />
                                The probability of King increased from 7.7% to 33.3%.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

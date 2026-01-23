import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, BookOpen, FileText, ArrowRight } from 'lucide-react';

export default function ConceptPanel() {
    const [activeGroup, setActiveGroup] = useState(0);

    // Analogy:
    // Queries = Students
    // Keys/Values = Textbooks/Notes
    // GQA = Multiple students sharing one textbook

    const groups = [
        {
            id: 0,
            color: "bg-orange-500",
            borderColor: "border-orange-500",
            textColor: "text-orange-600",
            key: { title: "Math Textbook", content: "Algebra & Calculus" },
            students: ["Student A", "Student B"]
        },
        {
            id: 1,
            color: "bg-blue-500",
            borderColor: "border-blue-500",
            textColor: "text-blue-600",
            key: { title: "History Book", content: "World Wars" },
            students: ["Student C", "Student D"]
        }
    ];

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-12">
                <h2 className="text-3xl font-bold text-fuchsia-600 dark:text-fuchsia-400 mb-4">The Study Group Analogy</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    <strong>Multi-Head Attention (MHA)</strong>: Every student has their own unique textbook. (Expensive!)
                    <br />
                    <strong>Multi-Query Attention (MQA)</strong>: All students share one single textbook. (Fast but lossy)
                    <br />
                    <strong>Grouped-Query Attention (GQA)</strong>: Students form <span className="text-fuchsia-600 font-bold">Groups</span>, and each group shares one textbook.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-16 w-full max-w-5xl">
                {/* Students (Queries) */}
                <div className="flex flex-col gap-8">
                    <h3 className="text-xl font-bold text-center text-slate-700 dark:text-slate-200">Students (Queries)</h3>
                    {groups.map((group, groupIndex) => (
                        <div key={group.id}
                            className={`p-6 rounded-2xl border-2 transition-all cursor-pointer ${activeGroup === groupIndex
                                    ? `${group.borderColor} bg-slate-50 dark:bg-slate-800/50 shadow-xl scale-105`
                                    : 'border-slate-200 dark:border-slate-700 hover:border-slate-400'
                                }`}
                            onClick={() => setActiveGroup(groupIndex)}
                        >
                            <div className="flex flex-wrap gap-4 justify-center">
                                {group.students.map((student, i) => (
                                    <motion.div
                                        key={i}
                                        className="flex flex-col items-center gap-2"
                                        initial={false}
                                        animate={{
                                            scale: activeGroup === groupIndex ? 1.1 : 1,
                                        }}
                                    >
                                        <div className={`p-3 rounded-full ${group.color} text-white`}>
                                            <Users size={24} />
                                        </div>
                                        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">{student}</span>
                                    </motion.div>
                                ))}
                            </div>
                            <div className={`mt-4 text-center text-sm font-bold ${group.textColor.replace('text-', 'text-opacity-80-')} uppercase tracking-wider`}>
                                Group {groupIndex + 1}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Textbooks (Keys/Values) */}
                <div className="flex flex-col justify-center gap-8">
                    <h3 className="text-xl font-bold text-center text-slate-700 dark:text-slate-200">Shared Resource (Key/Value)</h3>

                    <div className="relative h-[300px] flex items-center justify-center">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeGroup}
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.3 }}
                                className={`w-64 p-6 rounded-xl border-2 ${groups[activeGroup].borderColor} bg-white dark:bg-slate-800 shadow-2xl`}
                            >
                                <div className="flex items-center gap-4 mb-4">
                                    <div className={`p-3 rounded-lg ${groups[activeGroup].color} text-white`}>
                                        <BookOpen size={28} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-lg dark:text-white">KEY</h4>
                                        <p className="text-sm text-slate-500">{groups[activeGroup].key.title}</p>
                                    </div>
                                </div>
                                <div className="h-px bg-slate-200 dark:bg-slate-700 my-4" />
                                <div className="flex items-center gap-4">
                                    <div className={`p-3 rounded-lg ${groups[activeGroup].color} bg-opacity-20 text-${groups[activeGroup].color.replace('bg-', '')}`}>
                                        <FileText size={28} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-lg dark:text-white">VALUE</h4>
                                        <p className="text-sm text-slate-500">{groups[activeGroup].key.content}</p>
                                    </div>
                                </div>
                            </motion.div>
                        </AnimatePresence>

                        {/* Connection Lines (Visual only, implying connection) */}
                        <div className="absolute left-[-40px] top-1/2 -translate-y-1/2 md:block hidden">
                            <ArrowRight size={32} className="text-slate-300 dark:text-slate-600 animate-pulse" />
                        </div>
                    </div>

                    <div className="text-center p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                            The students in <strong>Group {activeGroup + 1}</strong> only need to load <strong>ONE</strong> textbook into memory.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

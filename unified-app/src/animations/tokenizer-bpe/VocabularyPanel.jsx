import React, { useState } from 'react';
import { Search, Hash, Type, Star, AlertCircle } from 'lucide-react';

function VocabularyPanel() {
  const [searchTerm, setSearchTerm] = useState('');
  const [showSpecial, setShowSpecial] = useState(true);

  // Sample CLIP vocabulary entries
  const clipVocab = [
    { id: 49406, token: '<|startoftext|>', type: 'special', desc: 'Beginning of sequence' },
    { id: 49407, token: '<|endoftext|>', type: 'special', desc: 'End of sequence' },
    { id: 0, token: '!', type: 'punctuation', desc: 'Exclamation mark' },
    { id: 256, token: 'the</w>', type: 'word', desc: 'Common word' },
    { id: 257, token: 'a</w>', type: 'word', desc: 'Common word' },
    { id: 320, token: 'a', type: 'subword', desc: 'Subword unit' },
    { id: 2368, token: 'cat</w>', type: 'word', desc: 'Animal' },
    { id: 1929, token: 'dog</w>', type: 'word', desc: 'Animal' },
    { id: 4044, token: 'beautiful</w>', type: 'word', desc: 'Adjective' },
    { id: 8619, token: 'photo', type: 'subword', desc: 'Part of photorealistic' },
    { id: 13440, token: 'real', type: 'subword', desc: 'Part of photorealistic' },
    { id: 2928, token: 'istic</w>', type: 'subword', desc: 'Suffix' },
    { id: 7270, token: 'sunset</w>', type: 'word', desc: 'Time/scene' },
    { id: 5765, token: 'painting</w>', type: 'word', desc: 'Art style' },
    { id: 1912, token: 'style</w>', type: 'word', desc: 'Descriptor' },
  ];

  const filteredVocab = clipVocab.filter(entry => {
    if (!showSpecial && entry.type === 'special') return false;
    if (searchTerm) {
      return entry.token.toLowerCase().includes(searchTerm.toLowerCase()) ||
             entry.id.toString().includes(searchTerm);
    }
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-2">Understanding Vocabulary</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          A tokenizer's vocabulary maps tokens (strings) to unique integer IDs.
          CLIP uses ~49,000 tokens; T5 uses ~32,000. Let's explore what's inside.
        </p>
      </div>

      {/* Vocabulary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-blue-500/10 rounded-xl p-4 text-center border border-blue-500/30">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">49,408</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">CLIP Vocab Size</div>
        </div>
        <div className="bg-green-500/10 rounded-xl p-4 text-center border border-green-500/30">
          <div className="text-2xl font-bold text-green-400">32,100</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">T5 Vocab Size</div>
        </div>
        <div className="bg-purple-500/10 rounded-xl p-4 text-center border border-purple-500/30">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">2</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Special Tokens</div>
        </div>
        <div className="bg-orange-500/10 rounded-xl p-4 text-center border border-orange-500/30">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">~40K</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Subword Tokens</div>
        </div>
      </div>

      {/* Search */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-800 dark:text-gray-400" size={18} />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search tokens or IDs..."
            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white focus:outline-none focus:border-orange-500"
          />
        </div>
        <label className="flex items-center gap-2 px-4 py-2 bg-black/30 rounded-lg cursor-pointer">
          <input
            type="checkbox"
            checked={showSpecial}
            onChange={(e) => setShowSpecial(e.target.checked)}
            className="w-4 h-4 rounded"
          />
          <span className="text-gray-700 dark:text-sm">Show special tokens</span>
        </label>
      </div>

      {/* Vocabulary Table */}
      <div className="bg-black/40 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-black/40">
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">ID</th>
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Token</th>
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Type</th>
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Description</th>
              </tr>
            </thead>
            <tbody>
              {filteredVocab.map((entry, i) => (
                <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                  <td className="py-3 px-4 font-mono text-yellow-400">{entry.id}</td>
                  <td className="py-3 px-4">
                    <code className={`px-2 py-1 rounded ${
                      entry.type === 'special' ? 'bg-red-500/30 text-red-300' :
                      entry.type === 'word' ? 'bg-green-500/30 text-green-300' :
                      entry.type === 'subword' ? 'bg-blue-500/30 text-blue-300' :
                      'bg-gray-500/30 text-gray-700 dark:text-gray-300'
                    }`}>
                      {entry.token}
                    </code>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      entry.type === 'special' ? 'bg-red-500/20 text-red-400' :
                      entry.type === 'word' ? 'bg-green-500/20 text-green-400' :
                      entry.type === 'subword' ? 'bg-blue-500/20 text-blue-600 dark:text-blue-400' :
                      'bg-gray-500/20 text-gray-800 dark:text-gray-400'
                    }`}>
                      {entry.type}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-400">{entry.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Token Types Explained */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-r from-red-500/10 to-orange-500/10 rounded-xl p-6 border border-red-500/30">
          <h3 className="font-semibold text-red-400 mb-3 flex items-center gap-2">
            <Star size={18} />
            Special Tokens
          </h3>
          <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <div className="bg-black/30 rounded p-3">
              <code className="text-red-400">&lt;|startoftext|&gt;</code> (49406)<br/>
              Marks the beginning of every sequence. Also called [BOS].
            </div>
            <div className="bg-black/30 rounded p-3">
              <code className="text-red-400">&lt;|endoftext|&gt;</code> (49407)<br/>
              Marks the end. The pooled embedding comes from here.
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-xl p-6 border border-green-500/30">
          <h3 className="font-semibold text-green-400 mb-3 flex items-center gap-2">
            <Type size={18} />
            Word vs Subword Tokens
          </h3>
          <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <div className="bg-black/30 rounded p-3">
              <code className="text-green-400">cat&lt;/w&gt;</code><br/>
              The <code>&lt;/w&gt;</code> suffix means "end of word". This is a complete word.
            </div>
            <div className="bg-black/30 rounded p-3">
              <code className="text-blue-600 dark:text-blue-400">photo</code><br/>
              No suffix = this continues into the next token. Part of a larger word.
            </div>
          </div>
        </div>
      </div>

      {/* Word Boundary Marker */}
      <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
        <h3 className="font-semibold text-yellow-400 mb-3 flex items-center gap-2">
          <AlertCircle size={18} />
          The &lt;/w&gt; Mystery
        </h3>
        <div className="text-gray-700 dark:text-gray-300 space-y-2 text-sm">
          <p>
            In CLIP's vocabulary, <code className="text-yellow-400">&lt;/w&gt;</code> marks word boundaries:
          </p>
          <div className="bg-black/40 rounded p-4 font-mono text-sm">
            <div className="text-gray-800 dark:text-gray-400"># "photorealistic sunset"</div>
            <div className="text-white">["photo", "realistic&lt;/w&gt;", "sunset&lt;/w&gt;"]</div>
            <br/>
            <div className="text-gray-800 dark:text-gray-400"># Meaning:</div>
            <div className="text-blue-600 dark:text-blue-400">"photo" → continues to next token</div>
            <div className="text-green-400">"realistic&lt;/w&gt;" → word ends here</div>
            <div className="text-green-400">"sunset&lt;/w&gt;" → standalone word</div>
          </div>
          <p className="text-gray-800 dark:text-xs mt-2">
            This allows reconstructing original spacing. Without it, we couldn't tell 
            "notebook" from "note book".
          </p>
        </div>
      </div>

      {/* Try It */}
      <div className="bg-black/30 rounded-xl p-6">
        <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-4">🔬 Try These Prompts</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-black/40 rounded p-4">
            <div className="text-orange-600 dark:text-orange-400 mb-2">"a cat"</div>
            <div className="text-gray-700 dark:text-gray-300">
              → [49406, 320, 2368, 49407]<br/>
              → [BOS] + "a" + "cat" + [EOS]<br/>
              = 4 tokens
            </div>
          </div>
          <div className="bg-black/40 rounded p-4">
            <div className="text-orange-600 dark:text-orange-400 mb-2">"photorealistic sunset"</div>
            <div className="text-gray-700 dark:text-gray-300">
              → [49406, 8619, 13440, 2928, 7270, 49407]<br/>
              → [BOS] + "photo" + "real" + "istic" + "sunset" + [EOS]<br/>
              = 6 tokens
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VocabularyPanel;

function escapeStr(value) {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r\n/g, '\n')
    .replace(/\n/g, '\\n');
}

function renderLeaf(node) {
  const fields = Object.entries(node.tip)
    .map(([key, value]) => `              ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const lesson = node.lessonId ? `\n            lessonId: '${node.lessonId}',` : '';
  const highlight = node.highlightTarget
    ? `\n            highlightTarget: { panel: '${node.highlightTarget.panel}', type: '${node.highlightTarget.type}' },`
    : '';
  return `          {
            id: '${node.id}',
            label: '${node.label.replace(/'/g, "\\'")}',
            tooltip: tip({
${fields}
            }),${lesson}${highlight}
          }`;
}

function renderBranch(branch) {
  const children = branch.children.map(renderLeaf).join(',\n');
  return `      {
        id: '${branch.id}',
        label: '${branch.label}',
        type: '${branch.type}',
        children: [
${children}
        ],
      }`;
}

export function renderMap(map) {
  const centerFields = Object.entries(map.center)
    .map(([key, value]) => `        ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const branches = map.branches.map(renderBranch).join(',\n');
  return `  '${map.id}': {
    center: {
      id: '${map.id}',
      label: '${map.label}',
      type: 'current',
      tooltip: tip({
${centerFields}
      }),
    },
    branches: [
${branches}
    ],
  }`;
}

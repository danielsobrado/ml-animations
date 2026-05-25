import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FlaskConical } from 'lucide-react';
import MindElixir, { SIDE } from 'mind-elixir';
import 'mind-elixir/style.css';
import { isConceptMap, NODE_TYPES } from '../../data/conceptMaps';
import ConceptTooltip from './ConceptTooltip';

const MIND_ELIXIR_THEME = {
  name: 'distill',
  type: 'light',
  palette: ['#234b8f', '#8a5a2b', '#4d6f39', '#7a3f36'],
  cssVar: {
    '--node-gap-x': '34px',
    '--node-gap-y': '16px',
    '--main-gap-x': '72px',
    '--main-gap-y': '24px',
    '--main-color': '#1f1f1f',
    '--main-bgcolor': '#f4efe5',
    '--main-bgcolor-transparent': 'rgba(244, 239, 229, 0.82)',
    '--color': '#1f1f1f',
    '--bgcolor': '#fbf8f1',
    '--selected': '#234b8f',
    '--accent-color': '#234b8f',
    '--root-color': '#1f1f1f',
    '--root-bgcolor': '#eee8dc',
    '--root-border-color': '#1f1f1f',
    '--root-radius': '0px',
    '--main-radius': '0px',
    '--topic-padding': '8px 12px',
    '--panel-color': '#1f1f1f',
    '--panel-bgcolor': '#fbf8f1',
    '--panel-border-color': '#d7ccb7',
    '--map-padding': '32px',
  },
};

function branchDirection(type) {
  return NODE_TYPES[type]?.side === 'left' ? MindElixir.LEFT : MindElixir.RIGHT;
}

function branchColor(type) {
  return NODE_TYPES[type]?.color || '#1f1f1f';
}

function branchTag(type) {
  return NODE_TYPES[type]?.label || type;
}

function tooltipText(tooltip) {
  if (!tooltip) return '';
  if (typeof tooltip === 'string') return tooltip;
  return [
    tooltip.short,
    tooltip.intuition,
    tooltip.example,
    tooltip.trap,
    tooltip.why,
  ].filter(Boolean).join(' ');
}

function toMindNode(node, branch, active = false) {
  return {
    id: `${branch}-${node.id}`,
    topic: node.label,
    direction: branch === 'prereq' ? MindElixir.LEFT : MindElixir.RIGHT,
    expanded: true,
    branchColor: active ? '#1f1f1f' : branch === 'prereq' ? '#234b8f' : '#8a5a2b',
    metadata: {
      lessonId: node.id,
      active,
      tooltip: node.explanation || node.description,
      kind: 'lesson',
    },
    tags: [active ? 'Current' : branch === 'prereq' ? 'Prereq' : 'Next'],
  };
}

function toInsightNode(node, index) {
  return {
    id: `insight-${node.id}`,
    topic: node.label,
    direction: index % 2 === 0 ? MindElixir.RIGHT : MindElixir.LEFT,
    expanded: true,
    branchColor: index % 2 === 0 ? '#4d6f39' : '#7a3f36',
    metadata: {
      tooltip: node.explanation,
      kind: 'insight',
    },
    tags: [node.tag || 'Note'],
  };
}

function conceptToMindNode(concept, branchType) {
  return {
    id: `${branchType}-${concept.id}`,
    topic: concept.label,
    direction: branchDirection(branchType),
    expanded: true,
    branchColor: branchColor(branchType),
    metadata: {
      tooltip: concept.tooltip,
      kind: 'concept',
      lessonId: concept.lessonId,
      highlightTarget: concept.highlightTarget,
      branchType,
    },
    tags: [branchTag(branchType)],
  };
}

function branchToMindNode(branch) {
  return {
    id: `branch-${branch.id}`,
    topic: branch.label,
    direction: branchDirection(branch.type),
    expanded: true,
    branchColor: branchColor(branch.type),
    metadata: {
      kind: 'branch',
      branchType: branch.type,
    },
    tags: [branchTag(branch.type)],
    children: branch.children.map((child) => conceptToMindNode(child, branch.type)),
  };
}

function makeLegacyMindmapData(mindmap) {
  return {
    nodeData: {
      id: `current-${mindmap.current.id}`,
      topic: mindmap.current.label,
      expanded: true,
      metadata: {
        lessonId: mindmap.current.id,
        active: true,
        tooltip: mindmap.current.explanation || mindmap.current.description,
        kind: 'lesson',
      },
      tags: ['Current'],
      children: [
        ...mindmap.prereqs.slice(0, 5).map((node) => toMindNode(node, 'prereq')),
        ...mindmap.insights.slice(0, 4).map((node, index) => toInsightNode(node, index)),
        ...mindmap.next.slice(0, 5).map((node) => toMindNode(node, 'next')),
      ],
    },
    direction: SIDE,
    theme: MIND_ELIXIR_THEME,
  };
}

function makeConceptMapData(map) {
  return {
    nodeData: {
      id: `current-${map.center.id}`,
      topic: map.center.label,
      expanded: true,
      metadata: {
        lessonId: map.center.id,
        active: true,
        tooltip: map.center.tooltip,
        kind: 'center',
      },
      tags: ['Current'],
      children: map.branches.map(branchToMindNode),
    },
    direction: SIDE,
    theme: MIND_ELIXIR_THEME,
  };
}

function makeMindmapData(mindmap) {
  if (isConceptMap(mindmap)) return makeConceptMapData(mindmap);
  return makeLegacyMindmapData(mindmap);
}

function collectMindmapTooltips(data) {
  const tooltips = new Map();
  const visit = (node) => {
    if (!node) return;
    const tooltip = tooltipText(node.metadata?.tooltip);
    if (tooltip) tooltips.set(node.id, tooltip);
    (node.children || []).forEach(visit);
  };
  visit(data.nodeData);
  return tooltips;
}

function findMindNode(root, nodeId) {
  if (!root || !nodeId) return null;
  if (root.id === nodeId) return root;
  for (const child of root.children || []) {
    const match = findMindNode(child, nodeId);
    if (match) return match;
  }
  return null;
}

function selectionFromNode(node) {
  if (!node?.metadata) return null;
  const { tooltip, lessonId, highlightTarget } = node.metadata;
  if (!tooltip && !lessonId) return null;
  return {
    label: node.topic,
    tooltip,
    lessonId,
    highlightTarget,
  };
}

export default function ConceptMindmap({ mindmap }) {
  const mapRef = useRef(null);
  const instanceRef = useRef(null);
  const navigate = useNavigate();
  const curated = isConceptMap(mindmap);
  const data = useMemo(() => makeMindmapData(mindmap), [mindmap]);
  const tooltipById = useMemo(() => collectMindmapTooltips(data), [data]);
  const [selection, setSelection] = useState(() => (
    curated ? selectionFromNode(data.nodeData) : null
  ));

  const currentLessonId = curated ? mindmap.center.id : mindmap.current.id;

  const resolveNodeId = useCallback((rawId) => {
    if (!rawId) return null;
    const normalized = rawId.replace(/^me/, '');
    if (findMindNode(data.nodeData, normalized)) return normalized;
    const withoutPrefix = normalized.replace(/^(?:prereq|next|insight|branch|prerequisite|mechanism|intuition|formula|trap|application)-/, '');
    const candidates = [
      normalized,
      `current-${withoutPrefix}`,
      `branch-${withoutPrefix}`,
      ...Object.keys(NODE_TYPES).flatMap((type) => [`${type}-${withoutPrefix}`]),
      `prereq-${withoutPrefix}`,
      `next-${withoutPrefix}`,
      `insight-${withoutPrefix}`,
    ];
    return candidates.find((candidate) => findMindNode(data.nodeData, candidate)) || normalized;
  }, [data.nodeData]);

  useEffect(() => {
    if (!curated) {
      setSelection(null);
      return;
    }
    setSelection(selectionFromNode(data.nodeData));
  }, [curated, data]);

  useEffect(() => {
    if (!mapRef.current) return undefined;

    const mind = new MindElixir({
      el: mapRef.current,
      direction: SIDE,
      editable: false,
      contextMenu: false,
      toolBar: false,
      keypress: false,
      mouseSelectionButton: 0,
      allowUndo: false,
      overflowHidden: true,
      theme: MIND_ELIXIR_THEME,
      scaleMin: 0.65,
      scaleMax: 1.2,
    });

    let disposed = false;
    const applyTooltips = () => {
      mapRef.current?.querySelectorAll('me-tpc').forEach((topic) => {
        const nodeId = topic.dataset?.nodeid || topic.getAttribute('data-nodeid');
        const normalizedId = resolveNodeId(nodeId);
        const tooltip = tooltipById.get(normalizedId);
        if (!tooltip) return;
        topic.setAttribute('title', tooltip);
        topic.setAttribute('aria-label', `${topic.textContent.trim()}. ${tooltip}`);
      });
    };

    const fitMap = () => window.requestAnimationFrame(() => {
      if (disposed || instanceRef.current !== mind) return;
      applyTooltips();
      mind.scaleFit();
      mind.toCenter();
    });

    mind.init(data);
    instanceRef.current = mind;
    fitMap();

    const getLessonId = (nodeId) => {
      const node = findMindNode(data.nodeData, resolveNodeId(nodeId));
      return node?.metadata?.lessonId;
    };

    const handleNodeClick = (event) => {
      const topic = event.target.closest?.('me-tpc');
      const rawId = topic?.dataset?.nodeid || topic?.getAttribute?.('data-nodeid');
      const nodeId = resolveNodeId(rawId);
      const node = findMindNode(data.nodeData, nodeId);
      if (!node) return;

      const lessonId = node.metadata?.lessonId;
      if (lessonId && lessonId !== currentLessonId) {
        event.preventDefault();
        navigate(`/animation/${lessonId}`);
        return;
      }

      if (curated) {
        const nextSelection = selectionFromNode(node);
        if (nextSelection) {
          event.preventDefault();
          setSelection(nextSelection);
        }
      } else {
        const legacyLessonId = getLessonId(rawId);
        if (!legacyLessonId || legacyLessonId === currentLessonId) return;
        event.preventDefault();
        navigate(`/animation/${legacyLessonId}`);
      }
    };

    mapRef.current.addEventListener('click', handleNodeClick, true);

    const handleResize = () => fitMap();
    window.addEventListener('resize', handleResize);

    return () => {
      disposed = true;
      window.removeEventListener('resize', handleResize);
      mapRef.current?.removeEventListener('click', handleNodeClick, true);
      mind.destroy();
      instanceRef.current = null;
    };
  }, [curated, currentLessonId, data, navigate, resolveNodeId, tooltipById]);

  return (
    <section
      className={['ua-concept-map', curated && 'ua-concept-map--curated'].filter(Boolean).join(' ')}
      aria-label="Concept map"
    >
      <div className="ua-learning-rail-head">
        <FlaskConical size={15} />
        <span>{curated ? 'Concept map' : 'Mindmap'}</span>
      </div>
      <div className={curated ? 'ua-concept-map-layout' : undefined}>
        <div ref={mapRef} className="ua-map-canvas" />
        {curated ? <ConceptTooltip selection={selection} /> : null}
      </div>
    </section>
  );
}

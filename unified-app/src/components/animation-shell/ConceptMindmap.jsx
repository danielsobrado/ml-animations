import React, { useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { FlaskConical } from 'lucide-react';
import MindElixir, { SIDE } from 'mind-elixir';
import 'mind-elixir/style.css';

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

function makeMindmapData(mindmap) {
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

function collectMindmapTooltips(data) {
  const tooltips = new Map();
  const visit = (node) => {
    if (!node) return;
    const tooltip = node.metadata?.tooltip;
    if (tooltip) tooltips.set(node.id, tooltip);
    (node.children || []).forEach(visit);
  };
  visit(data.nodeData);
  return tooltips;
}

export default function ConceptMindmap({ mindmap }) {
  const mapRef = useRef(null);
  const instanceRef = useRef(null);
  const navigate = useNavigate();
  const data = useMemo(() => makeMindmapData(mindmap), [mindmap]);
  const tooltipById = useMemo(() => collectMindmapTooltips(data), [data]);

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
        const normalizedId = nodeId?.replace(/^me/, '');
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
      const match = nodeId?.match(/^me(?:prereq|next)-(.+)$/);
      return match?.[1];
    };
    const handleNodeClick = (event) => {
      const topic = event.target.closest?.('me-tpc');
      const lessonId = getLessonId(topic?.dataset?.nodeid || topic?.getAttribute?.('data-nodeid'));
      if (!lessonId || lessonId === mindmap.current.id) return;
      event.preventDefault();
      navigate(`/animation/${lessonId}`);
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
  }, [data, mindmap.current.id, navigate, tooltipById]);

  return (
    <section className="ua-concept-map" aria-label="Concept mindmap">
      <div className="ua-learning-rail-head">
        <FlaskConical size={15} />
        <span>Mindmap</span>
      </div>
      <div ref={mapRef} className="ua-map-canvas" />
    </section>
  );
}

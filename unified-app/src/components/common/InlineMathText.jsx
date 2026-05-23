import React from 'react';
import Eq from '../../_design-system/Eq';

function isEscaped(text, index) {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function findClosingDollar(text, startIndex) {
  for (let index = startIndex; index < text.length; index += 1) {
    if (text[index] === '$' && !isEscaped(text, index)) {
      return index;
    }
  }
  return -1;
}

function splitInlineMath(text) {
  const tokens = [];
  let cursor = 0;

  while (cursor < text.length) {
    const dollarIndex = text.indexOf('$', cursor);
    const parenIndex = text.indexOf('\\(', cursor);
    const candidates = [dollarIndex, parenIndex].filter((index) => index >= 0);

    if (candidates.length === 0) {
      tokens.push({ type: 'text', value: text.slice(cursor) });
      break;
    }

    const nextIndex = Math.min(...candidates);
    const isDollar = nextIndex === dollarIndex;
    const closeIndex = isDollar
      ? findClosingDollar(text, nextIndex + 1)
      : text.indexOf('\\)', nextIndex + 2);

    if (closeIndex < 0) {
      tokens.push({ type: 'text', value: text.slice(cursor) });
      break;
    }

    if (nextIndex > cursor) {
      tokens.push({ type: 'text', value: text.slice(cursor, nextIndex) });
    }

    const startOffset = isDollar ? 1 : 2;
    tokens.push({ type: 'math', value: text.slice(nextIndex + startOffset, closeIndex) });
    cursor = closeIndex + startOffset;
  }

  return tokens;
}

export default function InlineMathText({ children }) {
  if (typeof children !== 'string') return children;

  const tokens = splitInlineMath(children);
  if (!tokens.some((token) => token.type === 'math')) return children;

  return (
    <>
      {tokens.map((token, index) => (
        token.type === 'math'
          ? <Eq key={`${token.value}-${index}`} tex={token.value} className="ua-inline-math" />
          : <React.Fragment key={`${token.value}-${index}`}>{token.value}</React.Fragment>
      ))}
    </>
  );
}

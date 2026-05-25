export const leaf = (id, label, tip, extra = {}) => ({ id, label, tip, ...extra });
export const branch = (id, label, type, children) => ({ id, label, type, children });

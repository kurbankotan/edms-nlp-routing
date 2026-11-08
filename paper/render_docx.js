// Renders paper/manuscript.json (blocks) into a Word document with docx-js.
// usage: node paper/render_docx.js paper/manuscript.json paper/Manuscript.docx
const fs = require("fs");
const path = require("path");
const docx = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
        ImageRun, PageBreak, BorderStyle, ShadingType, LevelFormat, TabStopType, PageNumber, Footer, Header, VerticalAlign } = docx;

const [,, inFile, outFile] = process.argv;
const M = JSON.parse(fs.readFileSync(inFile, "utf8"));
const FONT = "Times New Roman";
const BODY = 22;      // half-points (11 pt)
const SMALL = 18;     // 9 pt
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440;   // A4, 1-inch margins
const TEXT_W = PAGE_W - 2 * MARGIN;

// ---- inline markup: **bold**, *italic*, ^{sup}, _{sub}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), font: FONT, italics: true, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), font: FONT, superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), font: FONT, subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const para = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: opts.line ?? 276, before: opts.before ?? 0 },
  indent: opts.indent, keepNext: opts.keepNext, style: opts.style,
});

const children = [];
// ---- title block
children.push(new Paragraph({ children: runs(M.meta.title, { bold: true, size: 30 }), alignment: AlignmentType.LEFT, spacing: { after: 200, line: 300 } }));
children.push(new Paragraph({ children: runs(M.meta.authors, { size: BODY }), alignment: AlignmentType.LEFT, spacing: { after: 80 } }));
for (const a of M.meta.affiliations) children.push(new Paragraph({ children: runs(a, { size: 20 }), alignment: AlignmentType.LEFT, spacing: { after: 40 } }));
children.push(new Paragraph({ children: runs(M.meta.corresponding, { size: 20 }), alignment: AlignmentType.LEFT, spacing: { after: 240 } }));

let figN = 0, tabN = 0;
for (const b of M.blocks) {
  switch (b.type) {
    case "heading": {
      const lvl = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 }[b.level];
      children.push(new Paragraph({ text: b.text, heading: lvl, keepNext: true }));
      break;
    }
    case "paragraph":
      children.push(para(b.text, { indent: b.noindent ? undefined : undefined, after: b.after, run: b.run, align: b.align }));
      break;
    case "abstract":
      children.push(new Paragraph({ children: runs("Abstract", { bold: true, size: BODY }), spacing: { after: 80 } }));
      children.push(para(b.text, { run: { size: 20 }, line: 260 }));
      children.push(new Paragraph({ children: [new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: 20 }), ...runs(b.keywords, { size: 20 })], spacing: { after: 240 } }));
      break;
    case "highlights":
      children.push(new Paragraph({ children: runs("Highlights", { bold: true, size: BODY }), spacing: { after: 80 } }));
      for (const it of b.items) children.push(new Paragraph({ children: runs(it, { size: 20 }), numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 } }));
      children.push(new Paragraph({ text: "", spacing: { after: 120 } }));
      break;
    case "bullets":
      for (const it of b.items) children.push(new Paragraph({ children: runs(it), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } }));
      break;
    case "numbered":
      for (const it of b.items) children.push(new Paragraph({ children: runs(it), numbering: { reference: "numbers" + (b.id || ""), level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } }));
      break;
    case "equation":
      children.push(new Paragraph({
        children: [...runs(b.text, { italics: false, size: BODY }), new TextRun({ text: "\t" + b.label, font: FONT })],
        tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }], indent: { left: 720 }, spacing: { after: 120, before: 60 },
      }));
      break;
    case "figure": {
      figN += 1;
      const img = fs.readFileSync(path.resolve(path.dirname(inFile), "..", b.path));
      const dims = b.width_in ? { width: b.width_in * 96, height: b.width_in * 96 / b.aspect } : { width: 600, height: 400 };
      children.push(new Paragraph({ children: [new ImageRun({ data: img, type: "png", transformation: dims })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true }));
      children.push(para(b.caption, { run: { size: SMALL }, align: AlignmentType.LEFT, after: 200, line: 240 }));
      break;
    }
    case "table": {
      tabN += 1;
      children.push(para(b.caption, { run: { size: SMALL }, align: AlignmentType.LEFT, after: 60, line: 240, keepNext: true }));
      const n = b.columns.length;
      const widths = b.col_widths ? b.col_widths.map(w => Math.round(w * TEXT_W)) : Array(n).fill(Math.floor(TEXT_W / n));
      const fs_ = b.font_size || 16;
      const cell = (txt, i, head) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: head ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
        margins: { top: 30, bottom: 30, left: 60, right: 60 }, verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ children: runs(String(txt), { size: fs_, bold: head }), alignment: (i === 0 || head) ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 220 } })],
      });
      const rows = [new TableRow({ tableHeader: true, children: b.columns.map((c, i) => cell(c, i, true)) })];
      for (const r of b.rows) rows.push(new TableRow({ children: r.map((c, i) => cell(c, i, false)) }));
      const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
      const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
      children.push(new Table({ rows, columnWidths: widths, width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA },
        borders: { top: border, bottom: border, left: none, right: none, insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" }, insideVertical: none } }));
      if (b.notes) children.push(para(b.notes, { run: { size: 16 }, align: AlignmentType.LEFT, after: 200, line: 220, before: 40 }));
      else children.push(new Paragraph({ text: "", spacing: { after: 120 } }));
      break;
    }
    case "references":
      for (const r of b.items) children.push(new Paragraph({ children: runs(r, { size: 20 }), alignment: AlignmentType.LEFT, indent: { left: 567, hanging: 567 }, spacing: { after: 80, line: 240 } }));
      break;
    case "figure_missing":
      children.push(para(b.text, { run: { size: SMALL, italics: true }, align: AlignmentType.LEFT, after: 200 }));
      break;
    case "pagebreak":
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
  }
}

const doc = new Document({
  creator: "Kotan & Kırışoğlu", title: M.meta.title,
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers3", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outFile, buf); console.log("written", outFile, "figures:", figN, "tables:", tabN); });

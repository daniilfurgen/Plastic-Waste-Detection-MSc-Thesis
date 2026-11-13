// License: CC-BY-4.0
//
// Схема навчання Mean Teacher (рис. для розділу 2).
// Рендер: `typst compile mean_teacher.typ mean_teacher.pdf`
// Одна сторінка за розміром діаграми; у тезі підключається як `image("../figures/mean_teacher.pdf")`.

#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: cylinder, house, trapezium

#set page(width: auto, height: auto, margin: 1em)
#set text(font: "STIX Two Text", size: 10pt, lang: "uk")
#show math.equation: set text(font: "STIX Two Math")

// Палітра: світла заливка + темніший контур, щоб схема читалась і в чорно-білому друці.
#let c-lab = (fill: rgb("#dbe9f6"), stroke: rgb("#2b6ca3"))
#let c-unl = (fill: rgb("#fde6cc"), stroke: rgb("#c4690f"))
#let c-tea = (fill: rgb("#d9eed9"), stroke: rgb("#2e7d32"))
#let c-stu = (fill: rgb("#e6dcf2"), stroke: rgb("#6a3d9a"))
#let c-neu = (fill: luma(96%), stroke: luma(35%))

#let nbox(pal, pos, body, ..args) = node(
  pos,
  body,
  shape: rect,
  fill: pal.fill,
  stroke: 0.8pt + pal.stroke,
  corner-radius: 3pt,
  inset: 5pt,
  ..args,
)

#let data-edge = 0.8pt + luma(25%)
#let weight-edge = (
  paint: gradient.linear(rgb("#6a3d9a"), rgb("#2e7d32"), angle: -90deg),
  thickness: 1pt,
  dash: "dashed",
)
#let grad-edge = 1.1pt + rgb("#b3261e")

#let legend-item(stroke, body) = box(line(length: 9mm, stroke: stroke), baseline: -2pt) + h(2pt) + body + h(10pt)


#diagram(
  spacing: (2em, 2em),
  edge-stroke: data-edge,

  // ---------- Розігрів ----------
  nbox(c-lab, (-1.5, 1), name: <dl>, width: 7em, shape: cylinder.with())[Розмічена \ вибірка $cal(D)_l$],
  nbox(
    c-stu,
    (-1.5, 2),
    name: <warm>,
    shape: trapezium.with(dir: right, angle: 10deg),
    width: 7em,
    height: 3em,
  )[Учень $theta_0$],
  nbox(
    c-neu,
    (-1.5, 3),
    name: <init>,
    width: 7em,
  )[Ініціалізація \ $theta_T = theta_S = theta_0$ \ $hat(p)_0$],

  edge(<dl>, <warm>, "->"),
  edge(<warm>, <init>, "->"),

  // ---------- Крок навчання ----------
  nbox(c-unl, (0, 1.65), name: <ub>, width: 7em, shape: cylinder.with())[Нерозмічений \ пакет $cal(I)_u^t$],
  nbox(c-neu, (1, 1), name: <weak>)[Слабке \ подання],
  nbox(c-neu, (1, 3), name: <strong>)[Сильне \ подання],

  nbox(
    c-tea,
    (2, 1),
    name: <teacher>,
    shape: trapezium.with(dir: right, angle: 10deg),
    width: 7em,
    height: 3em,
  )[Вчитель $theta_T$],
  nbox(c-neu, (5, 1), name: <ssgt>, width: 7em)[Size-Stratified Ground Thresholding],
  nbox(c-unl, (5, 2.25), name: <pseudo>, width: 7em)[Псевдомітки \ $cal(L)_"pseudo"$],
  nbox(c-lab, (5, 3), name: <gt-labels>, width: 7em)[Істинні мітки \ $cal(L)_"ground truth"$],

  nbox(
    c-stu,
    (2, 3),
    name: <student>,
    shape: trapezium.with(dir: right, angle: 10deg),
    width: 7em,
    height: 3em,
  )[Учень $theta_S$],
  nbox(c-lab, (0, 3), name: <gt>, width: 7em, shape: cylinder.with())[Розмічений \ пакет $cal(I)_l^t$],

  // дані
  edge(<ub>, <weak>, "->", corner: right),
  edge(<ub>, (0, 2.2), (1, 2.2), <strong>, "->", corner: right),
  edge(<weak>, <teacher>, "->"),
  edge(<teacher>, <ssgt>, "->", label: text(fill: luma(35%))[_кандидати_], label-side: left),
  edge(<gt>, <strong>, "->"),
  edge(<strong>, <student>, "->"),
  edge(<ssgt>, <pseudo>, "->"),
  edge(<gt-labels>, <student>, "->"),
  edge(<pseudo>, (3.3, 2.25), (3.3, 3), <student>, "->"),

  // ваги
  edge(<student>, <teacher>, "-->", stroke: weight-edge, label: [EMA], label-side: right, label-pos: 0.45),

  // рамка кроку
  node(
    enclose: (<ub>, <weak>, <strong>, <teacher>, <ssgt>, <student>, <gt>),
    stroke: (dash: "dashed", paint: luma(45%), thickness: 0.7pt),
    corner-radius: 1em,
    inset: 1.5em,
    name: <loop>,
    align(top + left, text(
      size: 10pt,
      style: "italic",
      fill: luma(35%),
    )[Крок навчання]),
  ),
)

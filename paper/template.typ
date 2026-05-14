// LTeX: enabled=false
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *
#import "@preview/itemize:0.2.0" as el
#import "@preview/numbly:0.1.0": numbly

#let culine(text, sides: 1fr, left: auto, right: auto) = {
  let (left, right) = (left, right)
  if left == auto { left = sides }
  if right == auto { right = sides }
  let spacer = underline(extent: 10000cm)[#sym.space.nobreak]
  let hack(width) = box(width: width, clip: true, outset: (y: 1em), spacer)
  hack(left)
  underline(text)
  hack(right)
}

#let config(body) = {
  // Відступ між рядками
  let line-spacing = .65em

  // Налаштування сторінки
  set page(
    paper: "a4",
    margin: (left: 3cm, rest: 2cm),
    numbering: "1",
  )

  // Налаштування тексту
  set text(
    size: 14pt,
    font: ("Liberation Serif", "Times New Roman"),
    hyphenate: false,
    lang: "uk",
  )
  set par(
    justify: true,
    justification-limits: (tracking: (min: -0.01em, max: 0.02em)),
    leading: line-spacing,
    spacing: line-spacing,
    first-line-indent: (amount: 1cm, all: true),
  )
  set block(
    spacing: line-spacing,
  )

  // Налаштування списків
  show: el.paragraph-enum-list.with(indent: (1cm, auto))

  // Налаштування блоків з кодом
  show: codly-init.with()
  codly(languages: codly-languages)
  show raw: set text(size: 12pt, font: "Iosevka Slab")

  // Налаштування загловків
  show heading: set text(size: 14pt)
  show heading: set align(center)
  show heading: set block(spacing: line-spacing)
  set heading(numbering: numbly("Розділ {1}."))

  show heading.where(level: 1): smallcaps
  show heading.where(level: 3): set text(weight: "regular")
  show heading.where(level: 3): underline

  set outline(indent: 1em)
  show outline: set par(justify: true)
  show outline.entry.where(level: 1): smallcaps
  show outline.entry.where(level: 1): set text(weight: "bold")

  // Налаштування підписів
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set align(right)
  // show figure.caption.where(kind: table): set align(right)
  set figure.caption(separator: [. ])

  // Налаштування веб-посилань
  // show link: set text(fill: navy)
  // show link: underline

  // Математичні формули
  set math.equation(numbering: "(1)")
  show math.equation: set text(font: "STIX Two Math")

  // Виноски
  show footnote.entry: set text(size: 12pt)

  body
}

#let main-body-spacing(body) = {
  // Відступ між рядками
  let line-spacing = .65em * 1.5

  set par(leading: line-spacing, spacing: line-spacing)
  set block(spacing: line-spacing)
  show heading: set block(spacing: line-spacing)
  show heading.where(level: 1): set block(below: 2em)

  // Особливі терміни та симовли
  show "F1": $F_1$
  show "XX": highlight([XX (Замінити!)])

  body
}

#let abstract-paper-config(body) = {
  show: config.with()

  set page(numbering: none, margin: 2cm)
  set text(font: "Times New Roman", size: 15pt)
  set par(first-line-indent: (amount: .75cm, all: true))
  show: el.paragraph-enum-list.with(indent: (.75cm, auto))

  show heading.where(level: 1): set block(above: 2em)
  set heading(numbering: "1.")
  show heading: set text(size: 15pt)
  show heading: it => {
    set align(left)
    block(inset: (left: .75cm), it)
  }

  body
}

#let collection-paper-config(body) = {
  show: config.with()

  set page(numbering: "1", margin: 2cm, footer-descent: 0em)
  set text(font: "Times New Roman", size: 12pt)
  set par(first-line-indent: (amount: 1cm, all: true))
  show: el.paragraph-enum-list.with(indent: (1cm, auto))

  show heading.where(level: 1): set block(above: 2em)
  show heading: set text(size: 14pt)
  body
}

#let conference-paper-config(body) = {
  show: config.with()

  set page(numbering: none, margin: 2cm)
  set text(font: "Times New Roman", size: 14pt)
  set par(first-line-indent: (amount: 1.25cm, all: true), leading: .65em * 1.5, spacing: .65em * 1.5)
  set block(spacing: .65em * 1.5 * 2)
  show: el.paragraph-enum-list.with(indent: (1.25cm, auto))

  show heading.where(level: 1): set block(above: 2em)
  set heading(numbering: "1.")
  show heading: set text(size: 15pt)

  body
}

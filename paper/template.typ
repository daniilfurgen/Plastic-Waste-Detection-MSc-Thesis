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
    font: "Times New Roman",
    hyphenate: false,
    lang: "uk",
  )
  set par(
    justify: true,
    leading: line-spacing,
    spacing: line-spacing,
    first-line-indent: (amount: 1cm, all: true),
  )
  set block(
    spacing: line-spacing,
  )

  // Налаштування списків
  show: el.paragraph-enum-list.with(indent: 1cm)

  // Налаштування блоків з кодом
  show: codly-init.with()
  codly(languages: codly-languages)

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
  show figure.caption.where(kind: table): it => {
    align(right, it)
  }
  set figure.caption(separator: [. ])

  // Виноски
  show footnote.entry: set text(size: 12pt)

  body
}

#let main-body-spacing(body) = {
  // Відступ між рядками
  let line-spacing = .65em * 1.5

  set par(leading: line-spacing)
  set block(spacing: line-spacing)
  show heading: set block(spacing: line-spacing)
  show heading.where(level: 1): set block(below: 2em)

  // Особливі терміни та симовли
  show "F1": $F_1$
  show "XX": highlight([XX (Замінити!)])

  body
}

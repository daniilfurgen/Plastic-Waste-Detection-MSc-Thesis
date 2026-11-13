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

/// Таблиця у вигляді рисунка з необов'язковою приміткою під нею.
///
/// Решта іменованих і позиційних аргументів передається в `table`
/// (`columns`, `align`, `table.header(..)`, комірки). Мітку для посилань
/// додають після виклику: `#table-figure(..) <tab-name>`.
///
/// - caption (content): підпис таблиці.
/// - note (content, none): примітка під таблицею.
/// - placement (alignment, none): розміщення рисунка (`top`, `bottom`, `auto`).
/// - scope (str): `"column"` або `"parent"`.
#let table-figure(caption, note: none, placement: none, scope: "column", ..table-args) = figure(
  caption: caption,
  placement: placement,
  scope: scope,
  {
    set text(size: 0.85em)
    set par(justify: false)
    stack(
      spacing: .65em,
      table(
        fill: (x, y) => if y == 0 { luma(200) },
        row-gutter: (1mm, auto),
        ..table-args,
      ),
      ..if note != none {
        (
          {
            set align(left)
            set par(justify: true, first-line-indent: 0em)
            note
          },
        )
      },
    )
  },
)

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
  show figure.where(kind: table): fig => {
    set figure.caption(position: top)
    show figure.caption: set align(right)
    fig
  }
  set figure.caption(separator: [ --- ])

  show heading.where(level: 1): it => {
    it
    counter(math.equation).update(0)
    // counter(figure).update(0) does nothing
    // this is the correct default
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: raw)).update(0)
    counter(figure.where(kind: "algorithm")).update(0)
  }
  let prepend-with-section(numbering: "1.1", ..n) = context {
    let sec = counter(heading).get().first()
    std.numbering(numbering, sec, ..n)
  }
  set figure(numbering: prepend-with-section.with(numbering: "1.1"))

  // Посилання на рисунки, таблиці, алгоритми та формули. Номер розділу в `prepend-with-section`
  // береться з контексту, тобто з місця посилання, а не з місця об'єкта. Тому посилання з іншого
  // розділу отримувало б хибний префікс. Тут номер збирається за місцем самого об'єкта.
  show ref: it => {
    let el = it.element
    if el == none { return it }
    let is-figure = el.func() == figure
    if not is-figure and el.func() != math.equation { return it }
    context {
      let loc = el.location()
      let sec = counter(heading).at(loc).first()
      let object-counter = if is-figure { counter(figure.where(kind: el.kind)) } else { counter(math.equation) }
      let format = if is-figure { "1.1" } else { "(1.1)" }
      let number = std.numbering(format, sec, ..object-counter.at(loc))
      let supplement = if it.supplement == auto { el.supplement } else { it.supplement }
      link(loc, if supplement == none or supplement == [] { number } else { [#supplement~#number] })
    }
  }

  // Налаштування відступів рисунтів
  show figure: body => {
    set par(leading: line-spacing)
    set block(above: line-spacing * 3, below: line-spacing * 3)
    body
  }
  // Вирінвнювання в таблицях
  show table.cell: set align(horizon)

  // Налаштування веб-посилань
  // show link: set text(fill: navy)
  // show link: underline

  // Математичні формули
  set math.equation(numbering: prepend-with-section.with(numbering: "(1.1)"))
  show math.equation: set block(above: line-spacing * 3, below: line-spacing * 3)
  show math.equation: set text(font: "STIX Two Math")
  show math.equation.where(block: false): set math.frac(style: "horizontal")

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
  show heading: set block(above: line-spacing * 2, below: line-spacing * 2)

  // Особливі терміни та симовли
  show "F1": $F_1$
  show "mAP": $"mAP"$
  show "S3OD": [S#super[3]OD]
  show "mAP_S": $"mAP"_"S"$
  show "mAP_M": $"mAP"_"M"$
  show "mAP_L": $"mAP"_"L"$
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

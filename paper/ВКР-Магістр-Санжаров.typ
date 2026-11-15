// License: CC-BY-4.0

#import "template.typ": *

#set page(foreground: {
  place(
    left + horizon,
    dx: 1cm,
    rotate(-90deg, reflow: true)[
      #text(fill: luma(75%), size: 24pt, font: "Iosevka")[
        ЧЕРНЕТКА [#datetime.today().display("[day].[month].[year]")] / САНЖАРОВ Д.Р.
      ]
    ],
  )
})


#show: config.with()


// Титулка та вимоги
#include "sections/title.typ"

// Задання міжрядкового інтервалу приблизно 1.5
#show: main-body-spacing.with()

// Аннотація
#include "sections/abstract.typ"

// Зміст роботи
#outline()
#pagebreak()

// Загальний вміст
#include "sections/introduction.typ"

#include "sections/section-1.typ"

#include "sections/section-2.typ"

#include "sections/section-3.typ"

#include "sections/conclusions.typ"

#bibliography(
  "./sources.yaml",
  title: [Список використаних джерел],
  style: "dstu-8302-2015.csl",
)

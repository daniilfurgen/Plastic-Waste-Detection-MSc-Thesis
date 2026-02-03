#import "template.typ": *

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

#set heading(numbering: none)

= Висновки
#pagebreak()

#bibliography(
  "./sources.bib",
  title: [Список використаних джерел],
  style: "gost-r-705-2008-numeric", // Тимчасове рішення допоки не створено CSL конфігурацію для ДСТУ 8302:2015
)

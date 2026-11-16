// LTeX: language=uk

#set heading(numbering: none)

#heading(outlined: false)[Анотація]

У роботі досліджено напівкероване навчання детектора YOLO26 для виявлення малорозмірних пластикових відходів на поверхні річок і на морських узбережжях за обмеженої кількості анотованих даних. Реалізовано конвеєр Mean Teacher, у якому вчитель розмічає неанотовані кадри на кожному кроці, а поріг впевненості задається окремо для кожної категорії розміру об'єктів. Експерименти виконано на наборах FloW-img та UCWD за 10--100~% розмітки з контролем без псевдоміток. Приріст $"mAP"$ відносно контролю за 10~% розмітки становить +0,020 на FloW-img і +0,039 на UCWD, а за повної розмітки зменшується до +0,003 на FloW-img. Єдиним компонентом із очевидним внеском виявився повільний вчитель. На FloW-img YOLO26m досягає $"mAP"$ 43,9~% та $"mAP"_50$ 87,2~%, що за $"mAP"$ на рівні опублікованих результатів RT-DETR-X та ансамблю детекторів, при одній компактній моделі.

*Ключові слова:* детекція об'єктів; напівкероване навчання; Mean Teacher; YOLO26; пластикові відходи; малі об'єкти.

#heading(outlined: false)[Abstract]

The thesis studies semi-supervised training of the YOLO26 detector for finding small plastic waste on river surfaces and sea coasts when few annotated images are available. A Mean Teacher pipeline is implemented in which the teacher pseudo-labels unlabeled frames at every step and the confidence threshold is set separately for each object size category. Experiments were run on FloW-img and UCWD with 10--100~% of the labels against a control without pseudo-labels. At 10~% of the labels the gain in $"mAP"$ over the control is +0.020 on FloW-img and +0.039 on UCWD, and with all labels it shrinks to +0.003 on FloW-img. The slow teacher was the only component with a clear contribution. On FloW-img, YOLO26m reaches $"mAP"$ of 43.9~% and $"mAP"_50$ of 87.2~%, which in $"mAP"$ is on par with published results of RT-DETR-X and of a detector ensemble, with a single compact model.

*Keywords:* object detection; semi-supervised learning; Mean Teacher; YOLO26; plastic waste; small objects.

#pagebreak()

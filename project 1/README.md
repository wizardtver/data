
# <center> Анализ данных соискателей HH.ru </center>
## Оглавление
1. [Описание проекта](#Описание-проекта)
2. [Описание данных](#Описание-данных)
3. [Зависимости](#Зависимости)
4. [Установка проекта](#Установка-проекта)
5. [Использование проекта](#Использование-проекта)
6. [Авторы](#Авторы)
7. [Выводы](Использование-проекта)

## Описание проекта

> **Анализ данных соискателей HH.ru** – 
  Состоит из четырех основных этапов:
* базового анализа структуры данных.
* преобразования данных.
* Удаление признаков и записей, которые не несут полезной информации.
* проведения разведовательного анализа.

**Цель анализа данных с HH.ru** — избавиться от «мусора», который может помешать моделированию или исказить его результаты. Исследовать взаимосвязи между основными признаками резюме соискателей и уровнем желаемой заработной платы.

**Данный проект** направлен на демонстрацию взаимосвязей по основным признакам в построении модели определения уровня заработной платы.

**О структуре проекта:**
* [dst-3.0_16_1_hh_database](https://drive.google.com/drive/folders/1RkB9iM2t9xKzZbLCmUEWtvsxlZjtvVmk) - исходные данные
* [ExchangeRates](https://drive.google.com/drive/folders/1RkB9iM2t9xKzZbLCmUEWtvsxlZjtvVmk) - вспомогательные данные 
* [Project-1.Analys resumes HH.ipynb](https://github.com/wizardtver/data) - jupyter-ноутбук, содержащий основной код проекта, в котором демонстрируется решение по поставленным задачам
* [Grafics](https://github.com/wizardtver/data/tree/master/images) - статические графики plotly 


## Описание данных
В этом проекте используются данные HeadHunter (https://hh.ru/).

Требования HeadHunter состоят в построении модели, которая бы прогнозировала уровень желаемой зарплаты, опираясь на параметры соискателя, которые он указал о себе резюме.

Исходный датасет представляет собой набор данных и содержит 44744 обезличенные записи по 12 признакам:
Пол, возраст - пол и возраст соискателя;
ЗП - ожидаемая заработная плата;
Ищет работу на должность: - сведенья о желаемой должности;
Город, переезд, командировки - город проживания соискателя, его готовность к переезду и командировкам;
Занятость - желаемая занятость в виде одной из категорий: полная занятость, частичная занятость, проектная работа, волонтерство, стажировка;
График - желаемый график работы в виде одной из категорий: полный день, сменный график, гибкий график, удаленная работа, вахтовый метод;
Опыт работы - сведенья об опыте работы соискателя;
Последнее/нынешнее место работы - сведенья последнем/нынешнем месте работы;
Последняя/нынешняя должность - сведенья о последней/нынешней должности;
Образование и ВУЗ - уровень образования соискателя и наименование законченного учебного заведения;
Обновление резюме - дата и время последнего обновления резюме соискателем;
Авто - наличие у соискателя автомобиля. [здесь](https://drive.google.com/drive/folders/1RkB9iM2t9xKzZbLCmUEWtvsxlZjtvVmk).

## Используемые зависимости
* Python (3.11.3):
    * [numpy (1.24.3)](https://numpy.org)
    * [pandas (2.0.1)](https://pandas.pydata.org)
    * [matplotlib (3.7.1)](https://matplotlib.org)
    * [plotly (5.14.1)](https://plotly.org)
    * [seaborn (0.12.2)](https://seaborn.pydata.org)

## Установка проекта

```
git clone https://github.com/wizardtver/data
```

## Использование
Вся информация о работе представлена в jupyter-ноутбуке Project-1.Analys resumes HH.ipynb.

## Авторы

* [Andrey&SkillFactory]

## Выводы

Исходные данные были преобразованы посредством формирования новых информативных признаков, а так же удаления исходных, которые не несли в себе полезной информации. Очистка данных проводилась с помощью - удаления дублирующих записей, нивилирования пропусков, и учета выбросов. Исследование зависимостей выполнялось через визуализацию.[графики](https://github.com/wizardtver/data/tree/master/images) 
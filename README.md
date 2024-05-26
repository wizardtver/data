
# <center> Построение модели для оценки стоимости недвижимости в США </center>
## Оглавление
1. [Описание проекта](#Описание-проекта)
2. [Описание данных](#Описание-данных)
3. [Зависимости](#Зависимости)
4. [Установка проекта](#Установка-проекта)
5. [Использование проекта](#Использование-проекта)
6. [Авторы](#Авторы)
7. [Выводы](Использование-проекта)

## Описание проекта

> **Анализ данных** – 
  Состоит из следующих этапов:
* Преобразовать исходный датафрейм для дальнейшей работы.
* Удалить признаки, которые не несут полезной информации.
* Провести разведывательный анализ.
* Провести работу с пропусками.
* Добавить новые признаки.
* Выделить наиболее значимые признаки, влияющие на стоимость недвижимости.
> **Анализ моделей** – 
* Провести анализ моделей на выделенных признаках.
* Выбрать наиболее подходящую модель для решения поставленной задачи.
* Провести кроссвалидацию.
* Провести бутсрапвалидацию.
* Уменьшить модель для работы на сервере.
* Развернуть сервер.

**Цель работы состоит в создании быстрого и качественного минисервиса для оценки недвижимости в США** 

**Данный проект** 
 - направлен на демонстрацию всего спектра работ по изучению и анализу данных, созданию необходимых факторов для моделей, анализу моделей - для дальнейшего выбора быстрой и качественной модели, работающей на сервере.

**О структуре проекта:**
* [data.csv](https://drive.google.com/drive/folders/1AZwiskg9c_7CLv1Ov8ISAHneuD6eV4ri) - исходные данные
* [PRL_1.csv](https://drive.google.com/drive/folders/1AZwiskg9c_7CLv1Ov8ISAHneuD6eV4ri) - данные после предварительной обработки
* [PRL_address](https://github.com/wizardtver/data) - вспомогательные данные по геоточкам объектов
* [PRL_address_centr](https://github.com/wizardtver/data) - вспомогательные данные по геоточкам центров городов
* [PRL.ipynb](https://github.com/wizardtver/data) - jupyter-ноутбук, содержащий основной код проекта, в котором демонстрируется решение по поставленной задаче
* [XGBReg_model.pkl](https://drive.google.com/drive/folders/1AZwiskg9c_7CLv1Ov8ISAHneuD6eV4ri) - модель, развернутая на сервере


## Описание данных
В этом проекте используются данные агенства, занимающейся реализацией недвижимости в США.

Требования агенства состоят в построении сервиса, который бы прогнозировал стоимость недвижимости, опираясь на параметры, которые уже были проданы на рынке.

Исходный датасет представляет собой набор данных:
Статус недвижимости;
Наличие бассейна;
Тип недвижимости;
Адрес;
Количество санузлов;
Сведения о недвижимости;
Инфо о подключенной энергии;
Город;
Школы рядом с объектом;
Площадь;
Почтовый индекс;
Количество спален;
Штат;
Этажность;
Внутренняя информация агенства;
Стоимость объекта недвижимости.

## Установка необходимых зависимостей:
pip install -r requirements.txt

## Установка проекта

```
git clone https://github.com/wizardtver/data
```

## Использование
Вся информация о работе представлена в jupyter-ноутбуке.

## Авторы

* [Push&Gen]

## Выводы
Построенная модель отвечает всем требованиям агенства и предназначена для работы клиент-сервер.
# AIDAO24 (It's Just Business)

Добро пожаловать в репозиторий команды **It's Just Business**, участвовавшей в олимпиаде **AIDAO24**. В этом репозитории представлены решения двух заданий конкурса, в которых мы подробно описали наш подход, включая обработку данных, выбор моделей и настройку гиперпараметров.

*Welcome to the repository of the **It's Just Business** team, which participated in the **AIDAO24** competition. In this repository, we present solutions for two tasks from the competition, where we describe our approach, including data processing, model selection, and hyperparameter tuning.*

## Решение первого задания
*Solution for the first task*

### Обработка данных
В первом задании датасет был разделен на два подмножества в зависимости от наличия NaN значений. Каждое подмножество представляло собой данные с двух разных аппаратов. Для каждого подмножества мы извлекли следующие статистические признаки:
- Медиана
- Среднее значение
- Минимальное значение
- Максимальное значение
- Дисперсия и другие статистические показатели

*In the first task, the dataset was divided into two subsets based on the presence of NaN values. Each subset represented data from two different devices. For each subset, we extracted the following statistical features:*
- *Median*
- *Mean value*
- *Minimum value*
- *Maximum value*
- *Variance and other statistical metrics*

### Модели кластеризации
Мы протестировали несколько моделей кластеризации:
- **AgglomerativeClustering**
- **DBSCAN**
- **SpectralClustering**

В конечном итоге мы остановились на **SpectralClustering**, так как она показала наилучшие результаты. Мы экспериментировали с параметрами моделей, такими как количество кластеров, хотя в условиях задачи это значение было известно заранее.

*We tested several clustering models:*
- *AgglomerativeClustering*
- *DBSCAN*
- *SpectralClustering*

*In the end, we chose **SpectralClustering** because it showed the best results. We experimented with model parameters such as the number of clusters (even though the number of clusters was clearly specified in the task).*

### Результаты
Результат для каждого аппарата был сохранен отдельно, после чего оба предикта были объединены в один итоговый файл.

*The result for each device was saved separately, and then both predictions were combined into one final file.*

## Решение второго задания
*Solution for the second task*

### Извлечение признаков
Для второго задания мы также извлекли статистические признаки из исходных данных, чтобы улучшить качество кластеризации и классификации. Эти признаки включали основные статистические меры, такие как среднее, медиана и дисперсия.

*For the second task, we also extracted statistical features from the raw data to improve the quality of clustering and classification. These features included basic statistical measures such as mean, median, and variance.*

### Модель и настройка гиперпараметров
Мы остановились на использовании модели **LightGBM**, так как она показала отличное соотношение скорости и качества. Для поиска наилучших гиперпараметров использовалась технология grid search, подробности которой можно найти в файле `best_params_search`.

*We chose to use the **LightGBM** model because it provided an excellent balance between speed and quality. To find the best hyperparameters, we used grid search technology, the details of which can be found in the `best_params_search` file.*

### Проблемы и решения
Изначально мы планировали использовать нейронные сети, но из-за проблем с их проверкой на сайте контеста мы решили переключиться на модели классификации, такие как LightGBM.

*Initially, we planned to use neural networks, but due to issues with their evaluation on the competition website, we decided to switch to classification models like LightGBM.*

## Заключение
Ознакомиться с решениями поставленных задач можно, пройдя по соответствующим разделам в данном репозитории. Мы детально описали каждый этап разработки моделей и привели соответствующие скрипты, с которыми можно ознакомиться.

*You can familiarize yourself with the solutions to the tasks by exploring the respective sections in this repository. We have provided a detailed description of each step in the model development process, along with the corresponding scripts.*


Yours faithfully,

Frizen Daniil, Stepan Martynovich and Egor Bergner

<a href="https://github.com/guggenheimg">Frizen Daniil</a><br>


# Глибокий огляд літератури: frozen REVE, developmental EEG age prediction і зовнішній перенос

Дата пошуку: 2026-09-09  
Об’єкт огляду: питання з [`ARTICLE_SCOPE.md`](../../ARTICLE_SCOPE.md):

> Якщо encoder REVE заморожений, а preprocessing, checkpoint, HBN development split і seed-процедура зафіксовані, чи дають дедалі складніші age-prediction heads стабільний приріст над `mean_linear` і чи переносяться вони на зовнішню developmental EEG cohort?

## Висновок наперед

Я не знайшов опублікованої роботи, яка одночасно робить усі п’ять речей:

1. використовує саме frozen REVE як representation;
2. порівнює наперед задану драбину кількох heads при незмінному encoder і preprocessing;
3. перевіряє результат на кількох optimization seeds, а не на одному запуску;
4. не використовує зовнішню когорту для tuning, recalibration або вибору head;
5. робить sealed external developmental test на cohort, відмінній від HBN.

Отже, наша робота виглядає як вузьке методологічне дослідження стабільності frozen representations, а не як повторення однієї з відомих brain-age статей. Найближчі аналоги — це NeuralBench, STST-JEPA, NeuroAtlas, роботи з міждатасетною робастністю EEG brain-age і роботи про потужність зовнішньої валідації. Вони разом підтверджують, що питання є науково актуальним, але не закривають нашу конкретну комбінацію REVE + head-complexity ladder + multi-seed + sealed MIPDB.

REVE має принципове значення з двох різних причин. По-перше, він є самим representation, властивості якого ми тестуємо. По-друге, опублікований REVE pretraining inventory містить кілька HBN releases, тому HBN не можна описувати як encoder-independent test cohort. Це не робить дослідження некоректним, але змінює його правильне формулювання: HBN — development/evaluation cohort для probe над уже pretrained representation, а MIPDB — зовнішня перевірка переносу на іншу cohort. В опублікованому списку REVE MIPDB не названий, але це доказ із доступного inventory, а не криптографічна гарантія відсутності будь-якого pretraining overlap.

## Як проводився пошук

Пошук був сфокусований на шести перетинах, а не лише на словах `brain age`:

- developmental EEG age prediction, childhood/adolescence, brain maturation;
- HBN, MIPDB і зовнішня валідація між EEG cohorts;
- frozen probes, linear probing, attentive probing, representation transfer;
- REVE та EEG foundation models;
- nonlinear/complex heads, algorithmic variability, seed stability;
- dataset shift, inter-site generalization, power і reproducibility external validation.

Перевірялися первинні журнальні статті, PubMed/PMC, NeurIPS/OpenReview/arXiv, офіційні сторінки REVE і NeuralBench, а також dataset descriptors. Статті нижче розділені на: **прямі** (безпосередньо тестують частину нашого estimand), **найближчі** (покривають майже весь методологічний ризик, але з іншим encoder/data/protocol) і **контекстні** (пояснюють, чому age signal або external validation мають сенс).

Це максимально широкий цільовий пошук доступної індексованої літератури станом на дату вище. Він не є доказом, що не існує unpublished manuscript, thesis, workshop abstract або результату, який ще не індексується.

## 1. Найближчі прямі та методологічні аналоги

| Робота | Що саме вона робить | Чим корисна для нашого питання | Чого в ній немає |
|---|---|---|---|
| [REVE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html) | Self-supervised EEG encoder з 4D positional encoding для різних electrode layouts; понад 60 000 годин, 92 datasets, близько 25 000 subjects; демонструє transfer на 10 downstream tasks. | Дає саме representation і мотивацію перевіряти setup-agnostic перенос. | Оригінальна стаття не є age-prediction study: у paper/appendix немає прямого віку або MIPDB age evaluation. Вона не порівнює чотири frozen age heads і не робить HBN→MIPDB test. |
| [NeuralBench-EEG benchmark](https://arxiv.org/abs/2605.08495) і [офіційні frozen presets](https://facebookresearch.github.io/neuroai/neuralbench/full_benchmark.html) | Unified benchmark для 36 EEG tasks, 94 datasets, 14 models; має `linear_probe_flatten`, `linear_probe_mean`, `attentive_probe`, LoRA і full fine-tune. Age є окремим task. | Найближчий benchmark-прецедент: порівнює frozen encoder adaptation strategies і показує, що спосіб adaptation є окремою experimental variable. | У стандартному benchmark — три seeds (33–35), типові fixed splits і benchmark-defined metrics; немає нашої чотирьохголової ladder, 10 seeds, paired subject-level external deltas, sealed MIPDB або hierarchical bootstrap stability rule. |
| [STST-JEPA](https://arxiv.org/abs/2607.06629) | Self-supervised model на 47 703 sessions; frozen attentive probe для age дає на validation MAE 3.06 років і r=0.924 на 3 367 sessions; далі є легке fine-tuning. | Показує, що frozen attentive readout може витягувати developmental age signal із foundation representation. | Це одна probe architecture, а не head-complexity comparison; немає 10-seed paired inference і незалежного sealed external HBN→MIPDB age transfer. HBN також входить до pretraining mixture, тому HBN validation не є encoder-independent. |
| [NeuroAtlas](https://arxiv.org/abs/2605.14698) | Benchmark 20 models на 42 datasets і приблизно 260 000 годинах; frozen linear Ridge probes для brain-age across cohorts. Показує, що немає одного стабільного EEG-FM winner; mean pooling і regularization важливі; embedding dimensionality не передбачає performance. | Найсильніша сучасна підтримка нашої нульової/нестабільної гіпотези: складніший або більший representation не гарантує кращого age transfer, а простий regularized probe може бути стабільнішим. | Переважно sleep EEG, Ridge probe, інші cohorts і encoders; немає REVE, HBN/MIPDB design і чотирьох head families. |
| [Turgut et al., “Are foundation models useful feature extractors for EEG analysis?”](https://arxiv.org/abs/2502.21086) | Порівнює frozen linear probing і fine-tuning time-series foundation models на EEG age prediction у LEMON та інших tasks; використовує п’ять seeds. | Прямо ставить питання frozen feature extractor проти task-specific adaptation; показує, що linear probing може бути недостатнім для віку, а fine-tuning потрібен для частини tasks. | Не REVE, не developmental HBN→MIPDB transfer, не порівняння кількох frozen heads при зафіксованому encoder. |
| [Liu et al., “EEG Foundation Models: Progresses, Benchmarking, and Open Problems”](https://arxiv.org/abs/2601.17883) | Огляд 50 EEG foundation models і benchmark 12 open models на 13 datasets; порівнює linear probing, full fine-tuning і specialist models. | Систематизує broader evidence: linear probing часто недостатній, specialist models залишаються конкурентними, scale не гарантує generalization. | Це не age-specific REVE study і не external developmental validation. |
| [OpenEEGBench](https://github.com/braindecode/OpenEEGBench) | Сучасний benchmark parameter-efficient adaptation: frozen head, closed-form Ridge, LoRA, IA3, DoRA, full fine-tuning. | Підтверджує, що freeze-vs-adapt є самостійною benchmark-віссю, яку треба описувати окремо від encoder quality. | Наразі це general EEG benchmark без нашого HBN/MIPDB age protocol і без конкретної стабільності складних age heads. |
| [Rosenblatt et al., Nature Human Behaviour 2024](https://www.nature.com/articles/s41562-024-01931-7) | Понад 900 млн resampling simulations для external brain-phenotype validation; використовує HBN, ABCD, HCP-D, PNC та інші cohorts, зокрема age. | Показує, що external validation має дві sample-size dimensions і що external N критично впливає на power; підтримує нашу вимогу не трактувати один external score як остаточний. | Не EEG foundation model і не head comparison; це methodological evidence про validation power. |

## 2. Що відомо про developmental EEG age signal

### Vandenbosch et al. — стабільний EEG marker maturation

[Vandenbosch et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30609125/) використали 3-хвилинні eyes-closed EEG із NTR (`n=836`) і GNASA/WUSTL (`n=702`) у віці 5–18 років. На spectral power 1–24 Hz random forest досягав MAE близько 1.22 року; prediction error залишався стабільним через 1.5–2.1 року (`r≈.53–.74`), а оцінена heritability була 42–79%. Це важлива підстава вважати developmental age EEG signal біологічно осмисленим, а не лише acquisition artifact.

Водночас робота не відповідає нашому estimand: features handcrafted, model supervised і cohort-specific; немає frozen foundation representation, head ladder або sealed external transfer. Її правильна роль у related work — обґрунтувати наявність age signal, а не підтверджувати перевагу нелінійних probe над linear baseline.

### Iyer et al. — growth chart від infancy до adolescence

[Iyer et al. 2024](https://pubmed.ncbi.nlm.nih.gov/38537603/) побудували EEG growth chart на 1 056 типово розвинених дітей від 1 місяця до 18 років і перевірили 2-channel модель на незалежній cohort `n=723`. Res-NN досягав wMAE близько 0.85 року в основній setting і 2.27 року у 2-channel external setting. Це сильний precedent для вікової зовнішньої перевірки і водночас нагадування, що channel configuration і site shift можуть різко змінювати score.

Їхній model end-to-end і sleep-EEG-oriented, тому його не можна використовувати як доказ, що nonlinear head над REVE має стабільну перевагу. Але він підтримує саму логіку: developmental age model треба перевіряти поза training cohort.

### An et al. — infant/toddler age regression

[An et al. 2025](https://pubmed.ncbi.nlm.nih.gov/39721149/) вивчали 938 recordings від 457 дітей віком 2–38 місяців і порівнювали чотири linear/nonlinear ML models; MLP досягав `R²≈.83` і MAE близько 91.7 днів. Робота показує, що нелінійні моделі можуть допомогти в іншому віковому і physiological regime, але не показує, що вони стабільно перемагають linear head на замороженій general-purpose representation.

### Sun et al. і sleep brain-age line

Класична sleep-EEG brain-age line, започаткована [Sun et al., “Brain age from the electroencephalogram of sleep”](https://arxiv.org/abs/1805.06391), використовує інтерпретовану regularized linear/softplus architecture і великі adult sleep cohorts. Вона важлива як доказ, що прості regularized predictors можуть бути сильним baseline. Вона не є developmental frozen-REVE study, тому порівняння її чисел із нашим MIPDB результатом було б некоректним.

## 3. Зовнішній перенос, site shift і robustness

### Tveitstol et al. — найближчий brain-age generalization warning

[Tveitstol et al., IEEE TBME 2026](https://pubmed.ncbi.nlm.nih.gov/41329579/) протестували deep-learning brain-age models на п’яти EEG datasets з leave-one-dataset-out і leave-one-dataset-in evaluations, перебравши 1 805 hyperparameter configurations. Міждatasetні Pearson values коливалися приблизно від `.10` до `.84`, а `R²` у частині зовнішніх умов ставав сильно негативним. Автори показують, що population shift, hardware/acquisition shift і age-range mismatch можуть зруйнувати apparent within-dataset performance.

Це не REVE paper, але безпосередньо пояснює, навіщо нам MIPDB: HBN validation alone відповідає на питання доступності age signal у HBN, а не на питання переносимості head. За цією літературою external cohort є не «додатковим nice-to-have», а центральним stress test.

### Stevenson et al. — selective feature robustness across sites

[Stevenson et al. 2023](https://pubmed.ncbi.nlm.nih.gov/37442141/) навчали preterm-infant EEG age models на одному site і перевіряли на іншому. All-feature SVR погіршувався на external site (MAE 1.0 до 2.1), тоді як обмежений набір robust features переносився (MAE 1.0 до 1.1). Висновок релевантний для head comparison: більше feature capacity може підхопити site-specific structure і погіршити external transfer.

### Engemann et al. — benchmark до foundation-model era

[Engemann et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35905809/) створили reusable benchmark на чотирьох міжнародних M/EEG cohorts (`>2500` participants), порівнюючи classical і deep models. Spatially aware representations давали приблизно `R²=.60–.74`, а handcrafted + random forest були конкурентними і robust. Це важливе historical evidence: brain-age score залежить від representation і cohort, а deep model не автоматично кращий.

### HBN у зовнішній validation literature

Окремо [Rosenblatt et al.](https://www.nature.com/articles/s41562-024-01931-7) аналізують HBN у ширшому neuroimaging prediction context і показують, що маленькі external samples породжують inflated effects і false negatives. У нашому protocol `75` primary MIPDB subjects перевищують попередній minimum `50`, але це все одно не formal power guarantee. Тому hierarchical bootstrap intervals і bounded claim є правильнішими за висновок «head A точно кращий».

## 4. Model complexity, algorithmic variability і stability

### Ettling, Saba-Sadiya & Roig — різні algorithms знаходять різні patterns

[“Different Algorithms (Might) Uncover Different Patterns in EEG Brain-Age Prediction”](https://arxiv.org/abs/2402.09464) систематично порівнює algorithms/features для EEG brain-age і показує, що основні predictive patterns часто подібні, але частина висновків є model-specific. Це не probe study над REVE, однак прямо підтримує ідею не оцінювати складніший predictor лише за одним score: model capacity може змінювати не тільки accuracy, а й те, які patterns модель використовує.

### NeuroAtlas — complexity does not imply better age transfer

NeuroAtlas є найбільш близьким до нашої логіки сучасним результатом: на різних cohorts немає стабільного EEG foundation-model winner; linear Ridge з mean-pooled embeddings часто достатній; регуляризація стабілізує малі cohorts; dimensionality embedding не корелює систематично з age performance. Це узгоджується з нашим predeclared baseline-first підходом і робить нульовий результат на зовнішній cohort науково інформативним.

### BrainYears — nonlinear residual correction як окремий аналог

[BrainYears (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.03.26.714124v1.full) будує ElasticNet linear prediction, а потім gradient-boosted residual corrector. Це прямий приклад гіпотези «після dominant linear age axis може залишитися nonlinear residual structure». Але це adult functional-aging clock на іншій feature space і з іншим validation question; робота не доводить, що residual correction перенесеться між HBN і MIPDB або що він допоможе над frozen REVE.

## 5. HBN, MIPDB і попереднє поєднання цих datasets

[Langer et al. 2017](https://www.nature.com/articles/sdata201740) описують MIPDB як high-density task-free/task EEG resource з developmental age span. MIPDB є природною external cohort для нашого питання через іншу acquisition context і відмінну cohort construction.

Поєднання HBN і MIPDB вже зустрічається в іншому питанні: [Communications Biology 2024](https://www.nature.com/articles/s42003-024-06876-1) використовує HBN і MIPDB разом із multimodality brain-network features та SVM для age/development і diagnosis. Отже, сама пара datasets не є новою. Новизна нашого дизайну — не в тому, що ніхто раніше не відкривав MIPDB, а в тому, що ми використовуємо її як sealed external test для frozen REVE head comparison.

Для HBN age regression існують і практичні baselines, наприклад [EEG Dash age-regression example](https://eegdash.org/generated/auto_examples/applied/project_age_regression.html), де показано subject-aware split, Ridge на bandpower і sensitivity до site/equipment. Це корисний engineering baseline, але не peer-reviewed evidence про REVE або MIPDB transfer.

## 6. REVE-specific implications

### Що саме REVE додає до наукового питання

REVE важливий не лише як черговий pretrained model.

1. **Representation-level estimand.** Ми не питаємо, який end-to-end network найкраще регресує age. Ми питаємо, чи містить уже learned REVE embedding доступний, стабільний developmental age signal, який можна прочитати різними frozen heads.
2. **Heterogeneous EEG premise.** REVE використовує 4D positional encoding, щоб працювати з різними electrode arrangements. Це робить міжкоhortний transfer осмисленим, але не гарантує його: representation може кодувати acquisition/site identity разом із віком.
3. **Large pretraining makes leakage audit necessary.** Великі heterogeneous pretraining mixtures підвищують шанс на transfer, але водночас підвищують шанс overlap із downstream data. Тому REVE pretraining inventory — частина experimental interpretation, а не лише model citation.
4. **Original REVE paper leaves age transfer open.** У paper і appendix age/MIPDB не є прямим downstream claim. Наш experiment тому не просто повторює published REVE result; він перевіряє окрему, ще не закриту властивість representation.

### Найважливіше обмеження: HBN overlap

MIPDB не є підмножиною HBN і не є просто іншим release того самого dataset. Це окремий CMI/INDI resource зі своїм описом, release structure та participant identifiers. Обидві cohort пов’язані з developmental EEG і Child Mind ecosystem, але це не означає, що один dataset містить інший. Потенційний participant-level overlap не можна доводити лише назвами projects; для цього потрібен окремий provenance/identifier audit.

REVE appendix перелічує HBN-related OpenNeuro releases (`ds005505`, `ds005506`, `ds005507`, `ds005510`, `ds005511`, `ds005512`, `ds005514`) серед pretraining data. Це означає:

- HBN frozen-probe score не можна називати оцінкою на повністю unseen-for-encoder data;
- HBN є допустимим development cohort для питання «чи можна налаштувати head», але не чистим тестом representation generalization;
- MIPDB має особливу роль як cohort-level stress test поза HBN label/development workflow;
- навіть MIPDB не дає абсолютної гарантії pretraining independence без повного provenance audit усіх REVE training sources.

Правильне формулювання: **“external to the HBN development and head-fitting procedure”** або **“external developmental cohort evaluation under the published REVE source inventory”**. Сильніше формулювання “encoder-unseen external cohort” варто уникати, якщо немає повного підтвердження від авторів REVE.

## 7. Як наша робота стоїть щодо існуючих results

Місце нашого дослідження можна описати так:

- Vandenbosch, Iyer і An показують, що developmental age signal у EEG існує і може бути біологічно осмисленим.
- Engemann, Stevenson і Tveitstol показують, що within-cohort accuracy не гарантує міжsite/міжdataset transfer.
- NeuralBench, Turgut, STST-JEPA і OpenEEGBench показують, що frozen probing, attentive probing і fine-tuning — різні regimes; encoder quality не можна змішувати з adaptation quality.
- NeuroAtlas і EEG foundation-model benchmark 2026 показують, що більший або складніший foundation representation не гарантує стабільно кращого downstream score.
- Rosenblatt et al. пояснюють, чому external sample size і uncertainty мають бути частиною висновку.
- Наш протокол зводить ці застереження в один вузький test: **чи додає head complexity щось стабільне після того, як REVE, data processing, split, checkpoint і seed procedure вже заморожені?**

За цим визначенням наш потенційний результат “жоден складний head не встановив stable external gain” є повноцінним науковим результатом. Він не стверджує, що nonlinear heads ніколи не працюють; він стверджує, що у визначеній REVE/HBN/MIPDB/frozen-probe regime приріст не пройшов наперед заданий joint stability rule.

## 8. Рекомендована claim language для статті

Сильні, але коректні формулювання:

- “We test frozen age readout complexity over a fixed REVE representation.”
- “The primary question is stable external improvement, not the best single-seed score.”
- “MIPDB is external to HBN head fitting and is used only for sealed evaluation.”
- “No tested complex head established a stable external gain under the predeclared protocol.”
- “The result is bounded to the declared REVE checkpoint, four heads, HBN development split, MIPDB cohorts, preprocessing, and seeds 33–42.”

Формулювання, яких треба уникати:

- “REVE is validated for developmental age prediction” — оригінальна REVE paper цього не встановлює.
- “MIPDB proves encoder-level independence” — published source inventory дає evidence, але не абсолютну гарантію.
- “Nonlinear heads do not help” — наш result не покриває інші encoders, heads, datasets, fine-tuning або calibration.
- “No head is better” — confidence intervals crossing zero означають lack of established stable improvement, а не equivalence.
- “Our MIPDB score is an official NeuralBench score” — це окремий frozen-probe protocol; NeuralBench full fine-tuning є вторинною reproduction evidence.

## 9. Відтворюваність і зв’язок із поточним artifact set

Локальний study lock уже фіксує головні речі, яких зазвичай бракує в brain-age літературі:

- 4 наперед задані heads і REVE checkpoint;
- seeds 33–42;
- HBN representation cache та subject-level development split;
- sealed MIPDB pilot/primary/extrapolation allocation;
- відсутність MIPDB metrics у tuning або promotion decisions;
- 3 000 prediction records для 75 primary MIPDB subjects;
- paired seed-level comparisons, hierarchical bootstrap і Holm correction;
- bounded null interpretation замість claim equivalence.

Поточний audited result: baseline mean Pearson `0.644`; усі три candidate heads мали позитивні mean paired deltas, але всі 95% hierarchical-bootstrap intervals перетинали нуль. Це узгоджується з літературою про external shift і відсутність гарантованої переваги більш складного readout.

## 10. Основні джерела

- [REVE, NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html); [paper PDF](https://papers.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf).
- [NeuralBench arXiv](https://arxiv.org/abs/2605.08495); [official benchmark documentation](https://facebookresearch.github.io/neuroai/neuralbench/full_benchmark.html).
- [STST-JEPA](https://arxiv.org/abs/2607.06629).
- [NeuroAtlas](https://arxiv.org/abs/2605.14698).
- [Turgut et al., foundation models as EEG feature extractors](https://arxiv.org/abs/2502.21086).
- [Liu et al., EEG Foundation Models survey/benchmark](https://arxiv.org/abs/2601.17883).
- [OpenEEGBench](https://github.com/braindecode/OpenEEGBench).
- [Tveitstol et al., IEEE TBME 2026](https://pubmed.ncbi.nlm.nih.gov/41329579/).
- [Engemann et al., reusable brain-age benchmark](https://pubmed.ncbi.nlm.nih.gov/35905809/).
- [Vandenbosch et al., developmental EEG age](https://pubmed.ncbi.nlm.nih.gov/30609125/); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6865765/).
- [Iyer et al., EEG growth chart](https://pubmed.ncbi.nlm.nih.gov/38537603/); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11026939/).
- [An et al., infant/toddler EEG age prediction](https://pubmed.ncbi.nlm.nih.gov/39721149/).
- [Stevenson et al., inter-site generalizability](https://pubmed.ncbi.nlm.nih.gov/37442141/).
- [Ettling et al., algorithmic variability](https://arxiv.org/abs/2402.09464).
- [Rosenblatt et al., external validation power](https://www.nature.com/articles/s41562-024-01931-7).
- [Langer et al., MIPDB descriptor](https://www.nature.com/articles/sdata201740).
- [HBN/MIPDB multimodal development study](https://www.nature.com/articles/s42003-024-06876-1).
- [EEG Dash HBN age-regression example](https://eegdash.org/generated/auto_examples/applied/project_age_regression.html).
- [Stress-testing EEG foundation models with negative controls](https://arxiv.org/abs/2607.24519).
- [BrainYears nonlinear residual correction](https://www.biorxiv.org/content/10.64898/2026.03.26.714124v1.full).

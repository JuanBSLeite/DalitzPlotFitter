# Revisão de eficiência e backgrounds — 09/09/2026

Escopo: modelos funcionais/histogramas, SignalPDF, FitSession, MultiBackgroundNLL, categorias CP e compatibilidade com a geração de toys. Os achados descrevem a revisão inicial; as correções aplicadas estão registradas ao final.

## 1. Alta: mistura não-CP aceita frações e yields não físicos

`src/dalitzplotfitter/likelihood/mixture.py`, métodos `__post_init__`, `background_weights` e `__call__`.

A validação cobre a estrutura dos argumentos, mas não o domínio dos valores resolvidos. Não exige f_sig em [0,1], soma das frações explícitas <=1 nem yields não negativos. Além disso, densidades totais negativas/zero são substituídas por um piso positivo no logaritmo.

Reproduções com duas observações e densidades unitárias:

- f_sig=1.5: NLL=0, inclusive sob JIT com um Parameter resolvido no ponto de avaliação.
- Três backgrounds com frações explícitas 0.8 e 0.8: pesos [0.8,0.8,-0.6], NLL=0.
- Modo estendido com N_sig=3 e N_bg=-1: NLL=0.6137056388801094, finita.

Impacto: o minimizador pode aceitar pontos fora do espaço físico, mesmo quando a densidade total nas observações continua positiva. Limites individuais [0,1] não garantem que a soma das frações de múltiplos backgrounds seja <=1.

Recomendação: aplicar a mesma política já usada no CPJointNLL: rejeitar constantes iniciais inválidas, verificar valores resolvidos durante a minimização e retornar infinito para pontos inválidos, com proteção compatível com JAX/JIT/gradientes.

## 2. Alta: eventos fora do suporte recebem NLL finita

`src/dalitzplotfitter/workflow.py`, `_cached_signal_logpdf`; `src/dalitzplotfitter/pdf/signal.py`, `__call__`/`logpdf`; `MultiBackgroundNLL.__call__`.

O piso numérico transforma probabilidade exatamente zero em probabilidade positiva. Uma sessão de sinal puro contendo dez eventos explicitamente vetados retornou NLL=6907.755278982138 em float64, embora sua probabilidade seja zero. A reprodução usou um veto que reconhece os s12 dos dados e preserva a grade de normalização, isolando o problema de suporte de uma integral nula.

Impacto: uma seleção inconsistente entre dados e PDF pode seguir para o fit com um custo finito, sem diagnóstico claro. Com backgrounds, o critério relevante é a densidade total: sinal zero é permitido quando um background válido dá suporte ao evento.

Recomendação: distinguir zeros físicos de problemas de precisão. Para sinal puro, rejeitar dados fora do suporte ou retornar NLL infinita; na mistura, verificar positividade e finitude da densidade total. Não converter densidades negativas em valores positivos pelo piso.

## 3. Média: aceitação não-CP permite broadcasting e valores inválidos

`src/dalitzplotfitter/workflow.py`, `_acceptance`.

A multiplicação é feita antes de validar forma e domínio. Para três eventos, uma eficiência com shape (3,1) produz aceitação (3,3), em vez de falhar. Uma função bruta retornando [-1,-1,-1] também passa por esse helper sem erro. Isso pode causar arrays quadráticos, erros tardios na preparação do cache ou valores inválidos na NLL. Não foi demonstrado que a matriz (N,N) chega a um fit concluído; o defeito confirmado é a ausência de validação antes do broadcasting.

Os histogramas já validam bordas e valores finitos/não negativos; FunctionalEfficiency converte entradas inválidas em NaN. Essas proteções não abrangem todos os callables aceitos pela API. CPFitSession já faz a validação mais rigorosa.

Recomendação: aceitar somente escalar explicitamente expandido ou vetor (N,), validar finitude e não negatividade de eficiência e veto antes da multiplicação, tanto nos dados quanto na normalização. Eficiência relativa não precisa estar limitada a 1.

## 4. Média: background CP presente em uma única carga é recusado

`src/dalitzplotfitter/background/categories.py`, `CPBackgroundCategory.__post_init__` e `_validate_normalization`.

A categoria exige normalização estritamente positiva para cada carga separadamente. Entretanto, a PDF usa J_plus+J_minus como denominador conjunto, e uma carga pode ter taxa nula.

Reprodução: plus_values=[1,1], minus_values=[0,0], J_plus=1 e J_minus=0 produz ValueError: `onecharge minus normalization must be a positive finite scalar`.

Impacto: os toys CP corrigidos podem gerar um background apenas plus, mas a categoria de background do fit não consegue representar o mesmo caso.

Recomendação: aceitar integrais individuais finitas e não negativas, exigindo soma positiva e finita. Validar a consistência de uma integral nula com os valores fornecidos para aquela carga. Manter integral estritamente positiva para BackgroundCategory não-CP.

## Controles que passaram e convenções

- Com sinal NR em D+ -> pi- pi+ pi+ e eficiência 0.3+0.2*s12, a integral ponderada da SignalPDF na grade de normalização foi 1.0.
- Multiplicar essa eficiência por 7 alterou a densidade nos dados em no máximo 8.33e-17; cache e SignalPDF direta diferiram em no máximo 5.56e-17.
- Background constante teve normalização 5.063079718469924, igual à média dos pesos da amostra de integração, como esperado.
- A eficiência de sinal não é multiplicada automaticamente nos backgrounds: as shapes de background representam a distribuição observada; o veto é aplicado quando apply_veto=True. Isso é coerente entre workflow e toys e evita aplicar eficiência duas vezes a templates extraídos dos dados.
- Categorias pré-calculadas são usadas diretamente; seu autor precisa fornecer valores e integrais já correspondentes à seleção pretendida.
- Histogramas retornam alturas de bins, não convertem automaticamente contagens em densidade por área. Templates com bins de áreas diferentes precisam respeitar essa convenção.

## Validação executada

CPU, dois núcleos e JAX_ENABLE_X64=true:

`tests/test_multi_background.py`, `tests/test_cp_multi_background.py`, `tests/test_veto.py`, `tests/test_workflow.py`, `tests/test_cp_workflow.py`, `tests/test_cp_validation.py`: **34 passed em 12.45 s**.

As reproduções acima foram executadas separadamente. Os testes existentes verificam os caminhos válidos, mas não impediam os quatro problemas descritos. Não foi executada uma campanha estatística de fechamento com eficiência/background nesta revisão.


## Correções aplicadas

- MultiBackgroundNLL valida frações/yields iniciais e seus valores resolvidos no fit, incluindo o simplex e finitude da soma dos yields. Pontos inválidos retornam infinito por um ramo JAX protegido. Densidade total zero, negativa ou não finita não recebe piso positivo.
- SignalPDF preserva zeros físicos e seu logpdf retorna menos infinito fora do suporte. O caminho de sinal puro em FitSession usa a mesma avaliação segura dos logaritmos. O argumento `floor` foi mantido para compatibilidade de construção, sem recortar densidades. Backgrounds podem dar suporte onde o sinal é zero.
- FitSession valida forma, finitude e não negatividade de eficiência/veto antes de multiplicar, em dados e normalização. Escalares são expandidos explicitamente. SignalPDF também valida a forma e propaga entradas numéricas inválidas de modo compatível com JAX.
- CPBackgroundCategory permite integral zero em uma carga somente com valores nulos nessa carga, exigindo soma das integrais positiva e finita. A exigência de integral positiva no background não-CP permanece.
- Convenções e comportamento de suporte documentados em `docs/backgrounds_and_vetoes.md`.

Validação final: **59 passed em 16.40 s**, CPU/float64. Inclui os seis arquivos de testes da revisão e `tests/test_efficiency_background_regressions.py`, com casos de parâmetros inválidos sob JIT/gradientes, gradiente analítico em ponto válido, densidades float32, eficiência/veto de forma inválida, sinal fora do suporte e background CP de uma única carga dentro de CPJointNLL. Ruff passou nos módulos mixture, signal, categories e no novo arquivo de regressões; `git diff --check` passou.

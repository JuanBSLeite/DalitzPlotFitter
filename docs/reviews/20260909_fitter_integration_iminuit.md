# Revisão do fitter: integração, JAX e iminuit — 09/09/2026

Revisão do código local, sem alterações no núcleo e sem usar a GPU do teste 03. Os testes foram executados em processo separado, em CPU, limitado a dois núcleos. O escopo cobre o caminho single-sample de FitSession, caches de amplitudes, integração determinística e Minimizer; não constitui uma auditoria completa de CP, backgrounds, SCF ou todos os lineshapes.

## Achados confirmados

### 1. Alta prioridade: cache compartilhado reaproveita limites e valores padrão antigos

Em `src/dalitzplotfitter/fit/minimizer.py:121`, a assinatura considera nomes, flags fixed e valores fixos, mas não os valores, passos e limites dos parâmetros livres. O backend compartilhado inclui a tupla `free` do primeiro Minimizer. `_run` usa essa tupla para configurar o próximo Minuit.

Reprodução: uma mesma função `(x-3)**2`, primeiro com `Parameter('x', 0, bounds=(-5,5))`, depois com `Parameter('x', 0.5, bounds=(0,1))`. Basta preparar o primeiro backend antes de ajustar o segundo. O segundo ajuste retornou `valid=True`, limites `(-5,5)` e `x=2.999811927892423`, violando os limites solicitados. O valor padrão reaproveitado foi 0, em vez de 0.5.

O erro exige reaproveitar a mesma instância de função objetivo com declarações diferentes. Não explica por si só os mínimos secundários do teste 01, cujas declarações permanecem iguais, nem demonstra contaminação entre objetivos distintos no teste 03.

Correção recomendada: compartilhar apenas callbacks compilados e nomes; reconstruir `free` com os parâmetros do Minimizer atual. Alternativa mais simples, porém menos eficiente: incluir todas as definições na assinatura. Adicionar regressões para alteração de limites, passos e valor padrão.

### 2. Média prioridade: FitSession impede o reaproveitamento de normalização entre toys de dinâmica fixa

Em `src/dalitzplotfitter/workflow.py:202`, `signal_cache` passa sempre `acceptance_normalization`, inclusive quando é um vetor de uns. Em `src/dalitzplotfitter/decay.py:968`, o reaproveitamento exige `efficiency_normalization is None`.

Reprodução com modelo constante, grade Square Dalitz de resolução 10 e duas sessões de dez eventos: `_fixed_normalization_templates` permaneceu vazio nas duas sessões. Chamando `model.prepare_cache(data)` diretamente, o cache passou a conter um template.

Consequência: trabalho e alocações repetidos no teste 02, sem indicação de mudança do resultado numérico. Não aplicar automaticamente o mesmo reaproveitamento ao teste 03: sua dinâmica varia.

Correção recomendada: quando eficiência e veto forem ambos None, passar None para o cache. Testar o caminho público FitSession entre duas amostras, não apenas DecayModel.prepare_cache.

### 3. Controle de execução: ncall não é um orçamento total do fit

Em `src/dalitzplotfitter/fit/minimizer.py:358`, SIMPLEX e HESSE são chamados sem ncall; strategy=2 chama MIGRAD duas vezes, cada uma com o orçamento informado. Além disso, MIGRAD do iminuit tem sua própria política de repetição.

Em uma quadrática, `fit(ncall=1, simplex=True, hesse=True)` produziu 37 avaliações com strategy=1 e 51 com strategy=2. Não se trata apenas da pequena ultrapassagem de uma iteração: há etapas inteiras sem o orçamento informado. Isso importa em fits dinâmicos caros.

Documentar explicitamente que ncall se aplica a cada chamada MIGRAD, ou implementar um orçamento global com contabilização por etapa. A API do iminuit descreve ncall como limite aproximado por chamada: https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.migrad .

## Integração: estrutura coerente, com limitações importantes

- A convenção `mean(weights * f)` é coerente entre grids e matriz. Gauss–Legendre incorpora o jacobiano `4*m13*m23` e multiplica pesos pela quantidade de pontos retidos; `c† M c` representa a normalização com os conjugados na orientação adequada.
- O caminho dinâmico recalcula a normalização de cada componente livre e os blocos dinâmico–fixo e dinâmico–dinâmico. Os gradientes JAX incluem essas dependências. Os testes exercitam comparação com diferenças finitas, matriz direta e fechamento Asimov.
- **Precisão da grade não é garantida pela convergência do Minuit.** `DecayModel._adaptive_narrow_resonances` fixa o refinamento a partir dos parâmetros nominais. Isso é explícito no código, não uma falha oculta. Uma ressonância que se desloca ou se torna muito estreita pode sair da região bem resolvida. Os limites amplos do notebook 03 permitem larguras muito menores que a nominal; isso requer avaliação de precisão se o fit visitar essas regiões, sem que se possa afirmar viés nos resultados atuais apenas pela inspeção.
- O teste Asimov usa o mesmo suporte de integração para verdade e fit. Ele verifica consistência algébrica, mas não mede o erro da quadratura em relação à integral contínua. Para isso, comparar NLL, gradientes e parâmetros em grades progressivamente refinadas, idealmente também em outra parametrização de integração.
- **Memória por fit:** em `amplitude/cache.py:721`, a componente dinâmica é avaliada sobre toda a amostra de normalização; os valores fixos também ficam retidos para os termos cruzados. A opção `normalization_chunk_size` é documentada para o caminho de dinâmica fixa. Limpar sessões entre toys reduz retenção entre fits, mas não resolve o pico dessa avaliação e sua diferenciação. Uma melhoria estrutural é integrar blocos dinâmicos por chunks, com acumulação diferenciável dos pequenos blocos da matriz, verificando gradientes e memória do backward.

## Ligação com iminuit

A combinação NLL = -sum(log p), errordef=0.5, ordenação explícita de nomes e gradiente `jax.value_and_grad` está consistente no caminho revisado. O callback compartilha o valor e gradiente de um mesmo ponto. O ponto 1 precisa ser corrigido para que esse reaproveitamento não inclua configurações obsoletas.

`valid=True` é convergência local, não certificado de mínimo global nem de precisão da integração. HESSE avalia erros locais. Os mínimos alternativos observados no teste 01 não são, isoladamente, evidência de erro na ligação com iminuit.

## Validação

Primeiro grupo: **26 testes passaram** em 28,68 s:

- tests/test_minimizer.py
- tests/test_integration.py
- tests/test_gauss_legendre_integration.py
- tests/test_dynamic_fit_consistency.py
- tests/test_model_normalization_reuse.py

Reproduções adicionais dos achados 1–3 executadas separadamente. Nenhuma correção aplicada ao núcleo nesta revisão.

Segundo grupo: **20 testes passaram** em 10,99 s, cobrindo `tests/test_workflow.py` e `tests/test_amplitude_cache.py`. Total: **46 testes aprovados**. Incluem comparação de PDF genérica e cacheada, eficiência, blocos de múltiplas componentes dinâmicas e integração em chunks com último bloco parcial. A aprovação não cobre as regressões reproduzidas acima nem certifica a convergência da grade completa do teste 03.

## Correções aplicadas após a revisão

- Cache do Minimizer: callbacks compilados continuam compartilhados; parâmetros livres são reconstruídos a partir da instância atual. Regressão verifica limites, passo, valor padrão e start explícito, incluindo um mínimo fora dos novos limites.
- FitSession: sem eficiência/veto, passa None para permitir reaproveitar o template de normalização. Regressões verificam reutilização entre amostras e preservação do caminho com eficiência ou veto.
- ncall: mantida a semântica de limite aproximado por etapa, agora também passado a SIMPLEX e HESSE e documentado nos métodos fit. Não foi introduzido orçamento global.

Validação das alterações: 53 testes distintos aprovados nos sete arquivos listados acima. Na primeira execução, quatro testes novos usavam um mock que não calculava fval; esse problema do teste foi corrigido para envolver os métodos reais do iminuit. O arquivo de minimização foi então reexecutado: 16/16 aprovados; os demais 37 já haviam passado. `git diff --check` passou. O lint dos arquivos completos ainda aponta ocorrências preexistentes de estilo.

As limitações de precisão da grade para dinâmica variável e de memória por fit continuam como melhorias estruturais futuras; estas alterações não implementam integração dinâmica em chunks nem adaptação da grade durante o fit.

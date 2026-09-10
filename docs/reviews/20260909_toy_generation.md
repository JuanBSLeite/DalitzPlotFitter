# Revisão da geração de toys — 09/09/2026

Escopo: API de toys, accept-reject, inverse-transform/Rosenblatt, divisão CP, mistura de backgrounds e conservação dos quatro-momentos. Revisão e testes em CPU, em processos separados com dois núcleos. Os achados abaixo descrevem a revisão inicial; as correções aplicadas estão registradas ao final.

## 1. Alta prioridade: inverse-transform não preserva regiões vetadas

As CDFs e quantis são tabulados e interpolados continuamente. Platôs da CDF são reduzidos com `np.unique(..., return_index=True)`; interpolar a inversa através dos limites de regiões de densidade zero pode preencher os intervalos proibidos. A interpolação entre linhas condicionais e a aproximação trapezoidal também não garantem suporte exato em descontinuidades. A geração não verifica novamente a aceitação de cada evento produzido.

Reprodução com densidade constante em D+ -> pi- pi+ pi+, veto de 0.8 <= s12 <= 1.2 GeV², 50000 eventos, seed=112, usando diretamente DalitzInverseTransformSampler com densidade igual à função de aceitação:

| Resolução | Eventos dentro do veto |
|---|---:|
| 64 | 816 / 50000 |
| 256 | 280 / 50000 |
| 1024 (padrão) | 44 / 50000 |

Controle pela API accept-reject: zero violações em 2000 eventos, seed=23, pool_size=4096.

Impacto: eventos incompatíveis com o suporte da PDF ajustada, mesmo quando o usuário fornece veto. Aumentar a resolução reduz a discrepância neste exemplo, mas não assegura respeito exato ao veto.

Correção recomendada: tratar intervalos sem suporte na inversão e validar o suporte dos eventos finais; para vetos arbitrários, uma opção conservadora é usar accept-reject explicitamente. Apenas descartar eventos proibidos e repor a contagem não prova exatidão da densidade nas regiões permitidas: a aproximação da CDF ainda precisa de teste de convergência.

## 2. Alta prioridade: fallback accept-reject com dependência de momento perde a seleção angular

O fallback documentado para densidades que pedem p1/p2/p3 gera e aceita eventos com quatro-momentos. Porém, `toy_accept.py:454` executa `.without_momenta()` em todos os eventos selecionados. Ao final, `_attach_momenta` reconstrói orientações aleatórias a partir dos invariantes. Isso não preserva uma seleção que dependia da orientação original.

Reprodução: sinal constante, `efficiency=lambda d: d['p1'][:,3] > 0`, accept-reject, 2000 eventos, seed=24, pool_size=4096 e include_momenta=True. **1002 dos 2000 eventos retornados tinham p1_z <= 0**, violando a própria aceitação usada na geração.

Correção recomendada: preservar os momentos aceitos no fallback dependente de momentos. Manter a otimização de reconstrução apenas no caminho realmente dependente só de invariantes. Revisar também a mistura de componentes para não descartar os momentos selecionados quando alguns componentes usam o caminho compacto.

Esse problema não demonstra viés nos testes Genfit atuais sem seleção angular: afeta o fallback dependente de quatro-momentos.

## 3. Média prioridade: inverse-transform CP prepara uma carga de taxa zero

`toy_inverse.py:287` prepara sempre os dois samplers de sinal antes de verificar quantos eventos cabem a cada carga. Quando uma carga tem intensidade identicamente zero, a divisão binomial atribui corretamente zero eventos a ela, mas a preparação da CDF vazia falha.

Reprodução: componente NR com CPRealImag(1,0,1,0), logo c_plus=2 e c_minus=0; 100 eventos, seed=1. O inverse-transform falha com `inverse-transform target density has zero or invalid integral`; o accept-reject produz corretamente 100 eventos plus e zero minus.

Correção recomendada: preparar/gerar apenas componentes com contagem positiva, e validar separadamente que a integral conjunta é positiva e finita. Revisar pelo mesmo critério misturas com fração de sinal zero.

## Estrutura e limites numéricos

- O accept-reject inclui pesos de proposta nos scores, monitora estouros de envelope e descarta todos os eventos acumulados ao atualizar o envelope. Células sem scores positivos no piloto mantêm proposta positiva, evitando excluir regiões apenas por falta de amostragem inicial.
- O piloto e a monitoração não constituem uma prova de limite superior global para uma densidade arbitrária; picos não resolvidos merecem controles de estabilidade com tamanho do piloto e envelope_safety.
- O inverse-transform inclui o jacobiano `2*m12*(s13_max-s13_min)`, mas gera uma aproximação tabulada. A resolução das CDFs é independente da grade de normalização do fit. Um teste de fechamento de alta estatística precisa avaliar ambas as precisões.
- A geração CP calcula as probabilidades de carga pelas integrais aceitas e usa sorteio binomial. Misturas de backgrounds usam sorteio multinomial e validam frações relativas. Essas escolhas são coerentes com o modelo conjunto do fit, nas condições válidas.
- Sementes são registráveis e os toys retornam pesos unitários. Os testes existentes cobrem tamanho, reprodutibilidade, misturas, contagens CP, escrita ROOT e reconstrução de momentos on-shell/conservação. Esses controles não cobriam as três reproduções acima.

## Validação executada

- `tests/test_toy.py -k 'not can_write_one_root_tree'`: **16 passed, 1 deselected**, em 13.53 s.
- `tests/test_inverse_transform.py`: **2 passed**, em 1.33 s.
- Ambos executados em CPU com dois núcleos e `JAX_ENABLE_X64=true`. Isoladamente, os dois testes de inverse-transform falham em float32: suas tolerâncias de conservação são da ordem de 1e-10. Isso é uma dependência da configuração de precisão dos testes, distinta dos três bugs reproduzidos em float64.
- O teste de escrita/leitura ROOT ficou bloqueado também em processo separado. O diagnóstico após 60 s mostrou espera em `uproot.behaviors.TBranch._ranges_or_baskets_to_arrays`, durante `tree.arrays(["charge"], library="np")` (`tests/test_toy.py:297`). A execução foi interrompida; esse teste não está validado. O bloqueio ocorre na leitura do arquivo e não permite concluir, sozinho, que a geração esteja incorreta.


## Correções aplicadas

- A inversão da CDF marginal usa os intervalos completos, sem interpolar através dos platôs. Os extremos dos quantis também respeitam intervalos vazios iniciais/finais. Cada candidato é avaliado novamente na densidade original, já com a precisão dos invariantes retornados; candidatos de densidade zero são repostos. Após 100 lotes sem completar a amostra, a geração falha explicitamente. A interpolação condicional continua aproximada: respeitar o suporte não torna a densidade exata dentro dele.
- O accept-reject preserva os quatro-momentos aceitos no caminho dependente de momentos. Na mistura, reconstrói somente componentes compactos; a opção `include_momenta=False` remove momentos depois da seleção.
- O inverse-transform CP prepara somente cargas com contagem positiva. Ambos os métodos validam integrais de cargas finitas, não negativas e com soma positiva, e pulam a normalização do sinal em toys sem eventos de sinal. O gerador preparado também ignora sinal de fração zero e backgrounds de peso zero.
- `docs/toy_generation.md` descreve a checagem adicional da densidade, o custo correspondente e a necessidade de manter callbacks/modelo inalterados durante a reutilização das CDFs.

Reproduções originais após as correções: **zero violações de veto em 50000 eventos em cada resolução (64, 256, 1024)**; **zero violações angulares em 2000 eventos**; ambos os métodos CP retornam **100 plus e zero minus** no exemplo de taxa nula.

Validação: **35 testes aprovados, 1 desmarcado**, em CPU/float64, incluindo `tests/test_toy_regressions.py`, `tests/test_toy.py` e `tests/test_inverse_transform.py`. As regressões incluem reprodução da semente, suporte desconectado com comparação de populações contra integração independente, vetos pela API de sinal/mistura/CP, seleção angular em misturas, saída compacta, ambas as cargas de taxa nula, integral conjunta inválida e geração de backgrounds sem sinal. Ruff passou nos quatro arquivos Python alterados nesta correção. O teste ROOT permanece desmarcado pelo bloqueio de leitura documentado acima; nenhuma correção de I/O é reivindicada aqui.

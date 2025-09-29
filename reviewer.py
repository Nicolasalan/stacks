import os
import json
import google.generativeai as genai
from pypdf import PdfReader
import argparse

genai.configure(api_key="SUA_API_KEY_AQUI")

prompt_sistema_revisor = (
    "Você é um pesquisador de IA que está revisando um artigo científico submetido a uma conferência prestigiosa de ML. "
    "Seja crítico e cauteloso em sua decisão. Se um artigo é ruim ou você não tem certeza, dê notas baixas e rejeite."
)

instrucoes_template = """
Responda no seguinte formato:

PENSAMENTO:
<PENSAMENTO>

REVISÃO JSON:
```json
<JSON>
```

Em <PENSAMENTO>, primeiro discuta brevemente suas intuições e raciocínio para a avaliação.
Detalhe seus argumentos de alto nível, escolhas necessárias e resultados desejados da revisão.
Não faça comentários genéricos aqui, mas seja específico para o artigo atual.
Trate isso como a fase de anotações de sua revisão.

Em <JSON>, forneça a revisão em formato JSON com os seguintes campos na ordem:
- "Resumo": Um resumo do conteúdo do artigo e suas contribuições.
- "Pontos_Fortes": Uma lista dos pontos fortes do artigo.
- "Pontos_Fracos": Uma lista dos pontos fracos do artigo.
- "Originalidade": Uma nota de 1 a 4 (baixa, média, alta, muito alta).
- "Qualidade": Uma nota de 1 a 4 (baixa, média, alta, muito alta).
- "Clareza": Uma nota de 1 a 4 (baixa, média, alta, muito alta).
- "Significancia": Uma nota de 1 a 4 (baixa, média, alta, muito alta).
- "Questoes": Um conjunto de perguntas esclarecedoras a serem respondidas pelos autores.
- "Limitacoes": Um conjunto de limitações e potenciais impactos sociais negativos do trabalho.
- "Preocupacoes_Eticas": Um valor booleano indicando se há preocupações éticas.
- "Solidez": Uma nota de 1 a 4 (ruim, razoável, boa, excelente).
- "Apresentacao": Uma nota de 1 a 4 (ruim, razoável, boa, excelente).
- "Contribuicao": Uma nota de 1 a 4 (ruim, razoável, boa, excelente).
- "Nota_Geral": Uma nota de 1 a 10 (rejeição muito forte a qualidade de prêmio).
- "Confianca": Uma nota de 1 a 5 (baixa, média, alta, muito alta, absoluta).
- "Decisao": Uma decisão que deve ser uma das seguintes: Aceitar, Rejeitar.

Para o campo "Decisao", não use Aceitação Fraca, Aceitação Borderline, Rejeição Borderline, ou Rejeição Forte.
Use apenas Aceitar ou Rejeitar.
Este JSON será analisado automaticamente, então garanta que o formato seja preciso.
"""

formulario_neurips = f"""
## Formulário de Revisão
Abaixo está uma descrição das perguntas que você será solicitado no formulário de revisão para cada artigo.

1. Resumo: Resuma brevemente o artigo e suas contribuições.

2. Pontos Fortes e Fracos: Forneça uma avaliação completa dos pontos fortes e fracos do artigo:
   - Originalidade: As tarefas ou métodos são novos? É uma combinação nova de técnicas conhecidas?
   - Qualidade: A submissão é tecnicamente sólida? As alegações são bem suportadas?
   - Clareza: A submissão está claramente escrita? Está bem organizada?
   - Significância: Os resultados são importantes? Outros pesquisadores usarão as ideias?

3. Questões: Liste questões e sugestões para os autores.

4. Limitações: Os autores abordaram adequadamente as limitações e impacto social negativo?

5. Preocupações Éticas: Se há questões éticas, sinalize o artigo para revisão ética.

6. Solidez: Atribua uma nota numérica (1-4) indicando a solidez das alegações técnicas.
   4: excelente, 3: boa, 2: razoável, 1: ruim

7. Apresentação: Atribua uma nota numérica (1-4) para a qualidade da apresentação.
   4: excelente, 3: boa, 2: razoável, 1: ruim

8. Contribuição: Atribua uma nota numérica (1-4) para a qualidade da contribuição geral.
   4: excelente, 3: boa, 2: razoável, 1: ruim

9. Nota Geral: Forneça uma "nota geral" (1-10) para esta submissão.

10. Confiança: Forneça uma "pontuação de confiança" (1-5) para sua avaliação.

{instrucoes_template}
"""

def carregar_pdf(caminho_pdf):
    """Carrega texto do PDF usando pypdf"""
    try:
        leitor = PdfReader(caminho_pdf)
        texto = ""
        for pagina in leitor.pages:
            texto += pagina.extract_text() + "\n"

        if len(texto.strip()) < 100:
            raise Exception("Texto muito curto")

        return texto
    except Exception as e:
        print(f"Erro ao carregar PDF: {e}")
        return None

def extrair_json_entre_marcadores(texto):
    """Extrai JSON entre marcadores ```json```"""
    try:
        inicio = texto.find("```json")
        if inicio == -1:
            return None
        inicio += 7  # len("```json")

        fim = texto.find("```", inicio)
        if fim == -1:
            return None

        json_str = texto[inicio:fim].strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Erro ao extrair JSON: {e}")
        return None

def realizar_revisao(texto_artigo, modelo="gemini-1.5-flash", temperatura=0.7):
    """Realiza a revisão do artigo usando Gemini"""

    prompt_base = formulario_neurips + f"""
Aqui está o artigo que você deve revisar:
```
{texto_artigo}
```"""

    try:
        modelo_gen = genai.GenerativeModel(modelo)

        # Configuração de geração
        config_geracao = genai.types.GenerationConfig(
            temperature=temperatura,
            top_p=0.8,
            top_k=40,
            max_output_tokens=4000,
        )

        resposta = modelo_gen.generate_content(
            prompt_base,
            generation_config=config_geracao,
            system_instruction=prompt_sistema_revisor
        )

        revisao = extrair_json_entre_marcadores(resposta.text)

        if revisao is None:
            print("Falha ao extrair JSON da resposta do modelo")
            print("Resposta completa:")
            print(resposta.text)
            return None

        return revisao, resposta.text

    except Exception as e:
        print(f"Erro ao realizar revisão: {e}")
        return None, None

def salvar_revisao(revisao, texto_completo, caminho_saida):
    """Salva a revisão em arquivo JSON"""
    dados_saida = {
        "revisao": revisao,
        "texto_completo": texto_completo,
        "modelo": "gemini-1.5-flash"
    }

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(dados_saida, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Sistema de Revisão de Papers com Gemini')
    parser.add_argument('arquivo_pdf', help='Caminho para o arquivo PDF do artigo')
    parser.add_argument('--saida', '-o', default='revisao.json', help='Arquivo de saída da revisão')
    parser.add_argument('--modelo', '-m', default='gemini-1.5-flash', help='Modelo do Gemini a usar')
    parser.add_argument('--temperatura', '-t', type=float, default=0.7, help='Temperatura para geração')

    args = parser.parse_args()

    # Verificar se o arquivo PDF existe
    if not os.path.exists(args.arquivo_pdf):
        print(f"Erro: Arquivo {args.arquivo_pdf} não encontrado")
        return

    # Verificar se a API key está configurada
    if not genai.get_api_key():
        print("Erro: API key do Gemini não configurada")
        print("Configure sua API key editando a linha: genai.configure(api_key='SUA_API_KEY_AQUI')")
        return

    print(f"Carregando PDF: {args.arquivo_pdf}")
    texto_artigo = carregar_pdf(args.arquivo_pdf)

    if texto_artigo is None:
        print("Falha ao carregar o PDF")
        return

    print(f"Texto carregado: {len(texto_artigo)} caracteres")
    print("Realizando revisão com Gemini...")

    revisao, texto_completo = realizar_revisao(
        texto_artigo,
        modelo=args.modelo,
        temperatura=args.temperatura
    )

    if revisao is None:
        print("Falha ao realizar revisão")
        return

    print("Revisão concluída!")
    print(f"Salvando em: {args.saida}")

    salvar_revisao(revisao, texto_completo, args.saida)

    # Exibir resumo da revisão
    print("\n" + "="*50)
    print("RESUMO DA REVISÃO")
    print("="*50)
    print(f"Nota Geral: {revisao.get('Nota_Geral', 'N/A')}/10")
    print(f"Decisão: {revisao.get('Decisao', 'N/A')}")
    print(f"Confiança: {revisao.get('Confianca', 'N/A')}/5")
    print("\nPontos Fortes:")
    for ponto in revisao.get('Pontos_Fortes', []):
        print(f"• {ponto}")
    print("\nPontos Fracos:")
    for ponto in revisao.get('Pontos_Fracos', []):
        print(f"• {ponto}")

if __name__ == "__main__":
    main()

from crewai.tools import BaseTool
import pdfplumber


class LerHistorico(BaseTool):
    name: str = "Leitura de Histórico do Aluno"
    description: str = (
        "Lê arquivos em PDF no formato de Texto"
    )
    path: str = "/home/moab/IBM/enrollment-assistant/enrollment_assistant/knowledge/historico_cortado.pdf"

    def _run(self) -> str:
        conteudo = ""
        with pdfplumber.open(self.path) as pdf:
            for pagina in pdf.pages:
                conteudo += pagina.extract_text() + "\n"
        return conteudo

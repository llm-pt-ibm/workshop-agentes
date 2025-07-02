from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from enrollment_assistant.tools.custom_tool import LerHistorico
import os



from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class EnrollmentAssistant():
    """EnrollmentAssistant crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    optativas = CSVKnowledgeSource(
        file_paths=["disciplinas_optativas.csv"]
)
    fluxograma = JSONKnowledgeSource(
        file_paths=["fluxograma.json"]
)

    @agent
    def analista(self) -> Agent:
        return Agent(
            config=self.agents_config['analista'],
            verbose=True,
            tools=[LerHistorico()]
        )
    
    @agent
    def recomendador(self) -> Agent:
        return Agent(
            config=self.agents_config['recomendador'],
            verbose=True,
            knowledge_sources=[self.fluxograma, self.optativas],
            embedder={
                "provider": "google",
                "config": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "text-embedding-004" 
        }
    }
        )

    @task
    def extrair_perfil(self) -> Task:
        return Task(
            config=self.tasks_config['extrair_perfil'],
        )

    @task
    def recomendar_disciplinas(self) -> Task:
        return Task(
            config=self.tasks_config['recomendar_disciplinas'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the EnrollmentAssistant crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

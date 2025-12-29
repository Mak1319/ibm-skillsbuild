from typing import Dict , List 
from pydantic import BaseModel , Field
# from langchain_openai import ChatOpenAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os 

load_dotenv()

key = os.environ["GEMINI_KEY"]

class PreliminaryAgentState(BaseModel):
	professions: List[str]= Field(description="The profession the candidate belongs to", default_factory=list)
	organizations_worked : List[str] = Field(description= "The organizations the candidate worked before", default_factory=list)
	designation: List[str] = Field(description="The designations of the candidate worked", default_factory=list)
	experience: Dict[str, int] = Field(description="Work experience of the candidate for each designation with time in years", default_factory=dict)
	portfolio_length : int = Field(description="The length of the resume in pages" , default=1)

class CriticAgentState(BaseModel):
	negative_points: List[str]= Field(description="Negative points of the candidate", default_factory=list)

class FanAgentState(BaseModel):
	positive_points: List[str] =  Field(description="Positive points of the candidate", default_factory=list)

class ResumeCriticState(BaseModel):
	negative_points_resume: List[str]= Field(description="Negative points of the resume of the candidate", default_factory=list)

class ResumeFanState(BaseModel):
	positive_points_resume: List[str]= Field(description="Positive points of the resume of the candidate", default_factory=list)

class NeutralScoreJudge(BaseModel):
	scores_candidate_negative: List[int] = Field(description="The score for each negative points in sequence", default_factory=list)
	scores_candidate_positive: List[int]=  Field(description="The score for each positive point in sequence", default_factory=list)
	scores_resume_negative: List[int] = Field(description="The score for each negative point in sequence", default_factory=list)
	scores_resume_positive: List[int] = Field(description="The score for each positive point in sequence", default_factory=list)
	

class AgentState(PreliminaryAgentState, CriticAgentState, FanAgentState, ResumeCriticState, ResumeFanState, NeutralScoreJudge):
	resume: str = Field(description="The main resume of the candidate")
	def avg(self):
		return [ 
			sum(self.scores_candidate_negative) / len(self.scores_candidate_negative) ,
			sum(self.scores_candidate_positive) / len(self.scores_candidate_positive) ,
			sum(self.scores_resume_negative) / len(self.scores_resume_negative) ,
			sum(self.scores_resume_positive) / len(self.scores_resume_positive) 
		]
	def normalize_avg(self):
		return [ int( n * 100) for n in self.avg()]  

	def __or__(self, value):
        # PreliminaryAgentState fields
		if isinstance(value, PreliminaryAgentState):
			self.professions = value.professions
			self.organizations_worked = value.organizations_worked
			self.designation = value.designation
			self.experience = value.experience
			self.portfolio_length = value.portfolio_length
			return self

		# CriticAgentState fields
		if isinstance(value, CriticAgentState):
			self.negative_points = value.negative_points
			return self

		# FanAgentState fields
		if isinstance(value, FanAgentState):
			self.positive_points = value.positive_points
			return self
		# ResumeCriticState fields
		if isinstance(value, ResumeCriticState):
			self.negative_points_resume = value.negative_points_resume
			return self

		# ResumeFanState fields
		if isinstance(value, ResumeFanState):
			self.positive_points_resume = value.positive_points_resume
			return self
		if isinstance(value, NeutralScoreJudge):
			self.scores_candidate_negative = value.scores_candidate_negative
			self.scores_candidate_positive = value.scores_candidate_positive
			self.scores_resume_negative = value.scores_resume_negative
			self.scores_resume_positive = value.scores_resume_positive
			return self 

		return NotImplemented



llm = ChatGoogleGenerativeAI(
    model="models/gemini-3-flash-preview",
    temperature=0.7,
    google_api_key=key,
    response_mime_type="application/json"
)
memory = MemorySaver()



def preliminary_info(state: AgentState):
	parser = JsonOutputParser(pydantic_object=PreliminaryAgentState)
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			You are an HR manager in a well-known company. 
			Your work is to find out the basic info from the resume like: 
			1. Which profession the candidate belongs to or which profession the candidate wants to join.
			2. Which organizations the candidate has worked for before.
			3. Which designation the candidate has worked before or wants to work for.
			4. The experience in years for every designation.
			If any of the field is not available then do not make up any data on your own.
			Use empty string for string type, use 0 for int type where data is unavailable.
		"""),
		("user", "RESUME:{resume}\n\n{format_instructions}")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	response = chain.invoke({"resume": state.resume})
	response = PreliminaryAgentState.model_validate(response)
	state | response 
	return response

def critic_info(state: AgentState):
	parser = JsonOutputParser(pydantic_object=CriticAgentState)
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			You are the ultimate hater of the candidate.
			Your only job is to find out the negative points from the resume of the candidate.
			You have to ignore every positive point from the resume of the candidate.
			Do not make up any points on your own.
			You have to find minor to minor negative points of the candidate such that whatever happens to the world the candidate must not get a job.
		"""),
		("user", "RESUME:{resume}\n\n{format_instructions}")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	result = chain.invoke({"resume": state.resume})
	response = CriticAgentState.model_validate(result)
	state | response 
	return result

def fan_info(state: AgentState):
	parser = JsonOutputParser(pydantic_object=FanAgentState)
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			You are the ultimate fan of the candidate.
			Your only job is to find out the positive points from the resume of the candidate.
			You have to ignore every negative point from the resume of the candidate.
			Do not make up any points on your own.
			You have to find minor to minor positive points of the candidate such that whatever happens to the world the candidate must get a job.
		"""),
		("user", "RESUME:{resume}\n\n{format_instructions}")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	result = chain.invoke({"resume": state.resume})
	response = FanAgentState.model_validate(result)
	state | response 
	return result

def resume_critic(state: AgentState):

	# print(state)

	parser = JsonOutputParser(pydantic_object=ResumeCriticState)
	# profession = state.get('professions', ["a generic profession"])[0]
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			You are an HR manager of a company which is a global market leader in {profession}.
			Your job is to find out the best candidate for the post: {profession}.
			The user gives you a resume, and you have to find out the negative points from the resume which is written by the candidate.
			Avoid the negative points of the candidate. Just consider negative points of the resume writing.
		"""),
		("user", "RESUME:{resume}\n\n{format_instructions}")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	result = chain.invoke({"resume": state.resume , "profession": ", ".join(state.professions)})
	response = ResumeCriticState.model_validate(result)
	state | response 
	return result

def resume_fan(state: AgentState):
	parser = JsonOutputParser(pydantic_object=ResumeFanState)
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			You are an HR manager of a company which is a global market leader in {profession}.
			Your job is to find out the best candidate for the post: {profession}.
			The user gives you a resume, and you have to find out the positive points from the resume which is written by the candidate.
			Avoid the positive points of the candidate. Just consider positive points of the resume writing.
		"""),
		("user", "RESUME:{resume}\n\n{format_instructions}")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	result = chain.invoke({"resume": state.resume , "profession": ", ".join(state.professions)})
	response = ResumeFanState.model_validate(result)
	state | response 
	return result


def neutral_judge(state: AgentState):
	parser = JsonOutputParser(pydantic_object=NeutralScoreJudge)
	prompt = ChatPromptTemplate.from_messages([
		("system", """
			Assume you are a independent person. You were given a resume of a candidate. 
			The person have some good traits and some bad traits, and the resume you are provided have some 
			positive points and negative points , you have score this traits out of -10 to 10 for the profession {profession}
			You have to independent of biases. 
			You must read all the traits after that score the traits. 
			The scores must be in a order in the list.
		"""),
		("user", """
			positive points of the candidate: 
				{positive_points_candidate}
   
			negative points of the candidate: 
				{negative_points_candidate}
   
			positive points of the resume: 
				{positive_points_resume}
   
			negative points of the resume: 
				{negative_points_resume}
	
   
   		{format_instructions}""")
	])
	prompt = prompt.partial(format_instructions=parser.get_format_instructions())
	chain = prompt | llm | parser 
	result = chain.invoke({
		"positive_points_candidate":", ".join(state.positive_points) ,
		"negative_points_candidate":", ".join(state.negative_points) ,
		"positive_points_resume":", ".join(state.positive_points_resume) ,
		"negative_points_resume":", ".join(state.negative_points_resume) ,
		"profession": ", ".join(state.professions)})
	response = NeutralScoreJudge.model_validate(result)
	state | response 
	return result

def final_analyze(state: AgentState):
	# parser = JsonOutputParser(pydantic_object=NeutralScoreJudge)
	prompt = ChatPromptTemplate.from_messages([])
	chain = prompt | llm
	result = chain.invoke()
	return result 

workflow = StateGraph(AgentState)

workflow.add_node("informer", preliminary_info)
workflow.add_node("hater", critic_info)
workflow.add_node("fan", fan_info)
workflow.add_node("resume_critic", resume_critic)
workflow.add_node("resume_fan", resume_fan)
workflow.add_node("neutral_judge", neutral_judge)


workflow.add_edge(START, "informer")
workflow.add_edge( "informer", "hater")
workflow.add_edge( "hater" , "fan")
workflow.add_edge("fan", "resume_critic")
workflow.add_edge("resume_critic", "resume_fan")
workflow.add_edge("resume_fan", "neutral_judge")
workflow.add_edge("neutral_judge", END)

app = workflow.compile(checkpointer=memory)
import re
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, BeforeValidator, ValidationError, field_validator

from social_groups.trialrunner.config import BackendInfo, LLMConfig, get_llm
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
)
from social_groups.trialrunner.decision_schemes.thinking_mad import (
    strip_out_thinking_process,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import retrieve_single_answer_info
from social_groups.trialrunner.utils.phoenix import phoenix_log_span
from social_groups.trialrunner.utils.tool_calls import (
    InvalidToolCallException,
    NoToolCallsException,
    parse_tool_call_arguments,
)


class Agent(BaseModel):
    llm: LLMConfig
    backend: BackendInfo


class TribalCouncilConfiguration(BaseModel):
    proposal_agent: Agent
    council_agent: Agent
    council_size: int
    num_different_proposals: int
    accept_after_n_invalid_proposals: int
    total_invalid_question_formulations_accepted: int
    total_invalid_final_decisions_accepted: int


def extract_letter(v: str):
    if isinstance(v, str):
        # TODO: make this configurable for different question types:
        match = re.search(r"(?i)[\(\[]?([A-J])[\)\]\.:]?\b", v.strip())
        if match:
            return match.group(1).upper()
    raise ValidationError(f"Cannot extract A–J choice from: {v!r}")


class Proposal(BaseModel):
    # TODO: make this configurable for different question types:
    answer = Annotated[
        Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        BeforeValidator(extract_letter),
    ]
    reasoning: str


class Reaction(BaseModel):
    question: str
    reactions: list[str]


PROPOSAL_SYSTEM_PROMPT = (
    "You are an expert, professional professor from a renowned university."
    "You get the following question and have to answer it"
    "in front of a committee while providing extensive explanation on why your answer is correct."
    "Use the given tool to propose in front of the committee after you though about the problem."
)

QUESTIONER_SYSTEM_PROMPT = (
    "You are part of a council which retrieves answers to extremely important questions."
    "You see proposals of answers to a given question including reasoning and you have to ask a question to the proposers"
    "which will help distinguish the answers and lead to correctly finding the correct proposal."
    "The questions should be thought provoking, direct and help distinguish between the given proposals and very direct."
    "Please formulate them about specific parts, assumptions or facts of the given proposals."
)


def make_proposal(llm: BaseChatModel, question: str) -> None | Proposal:
    @tool(return_direct=True)
    def propose_solution(correct_answer: str, reasoning: str):
        """
        Submit the answer to the question that you think is correct.
        Please provide extensive reasoning on why this is correct,
        which assumptions and knowledge was used to retrieve the answer
        and what argumentation is needed to come to the conclusion.

        :arg correct_answer: The correct answer-letter of the question. in the form: (X)
        :arg reasoning: Fully specified reasoning path to retrieve this answer.
        """
        pass

    ai_msg = llm.bind_tools([propose_solution]).invoke(
        [
            SystemMessage(PROPOSAL_SYSTEM_PROMPT),
            HumanMessage(question),
            HumanMessage(
                "You have to first think about the correct answer, then you must submit it using the given tool!"
            ),
        ]
    )

    args = parse_tool_call_arguments(ai_msg)
    try:
        return Proposal(answer=args["correct_answer"], reasoning=args["reasoning"])  # noqa: some wired behaviour with validator

    except (ValidationError, KeyError) as e:
        raise InvalidToolCallException() from e


def trim_answers(q: str):
    return q[3:].split("\n")[0]


def get_answer(original_question: str, answer: str):
    results = (
        original_question[3:].strip().split("Options are:")[-1].strip().split("\n")
    )
    answers = {}
    for result in results:
        letter, word = result.strip()[1:].split("):")
        answers[letter] = word

    return answers[answer].strip()


def form_question_about_proposal(
    llm: BaseChatModel,
    original_question: str,
    proposals: list[Proposal],
    reactions: list[Reaction],
) -> str:
    @tool(return_direct=True)
    def propose_question(question: str):
        """
        Use this tool to submit your question to the committee.
        Please provide a very thoughtful question after you analyzed the proposals
        and former conversation.
        """
        pass

    ai_msg = llm.bind_tools([propose_question]).invoke(
        [
            SystemMessage(QUESTIONER_SYSTEM_PROMPT),
            AIMessage(f"The original question is: {trim_answers(original_question)}"),
        ]
        + [
            HumanMessage(
                f"Proposal {i}: {p.reasoning}. Therefore the answer should be {get_answer(original_question, p.answer)}"
            )
            for i, p in enumerate(proposals)
        ]
        + [
            x
            for r in reactions
            for x in [
                AIMessage(r.question),
                *[
                    HumanMessage(f"Proposal {i}: {a}")
                    for i, a in enumerate(r.reactions)
                ],
            ]
        ]
        + [
            HumanMessage(
                "What is your question to the proposers? Do not repeat any question asked before."
                "Please accompany them sense full. You have to finally submit using the tool provided."
            )
        ]
    )
    try:
        return parse_tool_call_arguments(ai_msg)["question"]
    except KeyError as e:
        raise InvalidToolCallException() from e


def answer_question_about_proposal(
    proposal_llm: BaseChatModel,
    original_question: str,
    proposals: list[Proposal],
    old_reactions: list[Reaction],
    question_about_proposal: str,
) -> Reaction:
    # @tool
    # def answer_the_question(answer: str):
    #     """
    #     Provide an official answer to the question.
    #     Use this tool AFTER you analyzed your proposal again, thought about implications
    #     and evaluated the other proposals.
    #     THEN carefully answer the question, you can change your original opinion if
    #      you think that it is necessary.
    #     """
    #     pass

    new_reactions: list[str] = []

    for current_proposal in range(len(proposals)):
        messages = (
            [
                SystemMessage(
                    "You are an expert, professional professor from a renowned university."
                    "You get the following question and have to answer it."
                    "in front of a committee while providing extensive explanation on why your answer is correct."
                    "\n After you got questions from the committee you have to answer it as good as you can"
                    "to either defend your original proposed solution or get convinced from another answer."
                    "Please adapt your answer to what you think is correct. \n"
                    "You have to submit your answer to the question using the given tool."
                    "USe the tool AFTER you though about the question again and revisited your assumptions."
                ),
                HumanMessage(original_question),
            ]
            + [
                HumanMessage(
                    f"Proposal {i}: The correct answer is ({p.answer}) {get_answer(original_question, p.answer)}. {p.reasoning}"
                )
                if i != current_proposal
                else AIMessage(
                    f"Proposal {i}: The correct answer is ({p.answer}) {get_answer(original_question, p.answer)}. {p.reasoning}"
                )
                for i, p in enumerate(proposals)
            ]
            + [
                x
                for r in old_reactions
                for x in [
                    HumanMessage(r.question),
                    *[
                        HumanMessage(f"Proposal {i}: {a}")
                        if i != current_proposal
                        else AIMessage(f"Proposal {i}: {a}")
                        for i, a in enumerate(r.reactions)
                    ],
                ]
            ]
            + [HumanMessage(question_about_proposal)]
            + [HumanMessage(f"Proposal {i}: {r}") for i, r in enumerate(new_reactions)]
        )

        # ai_msg = proposal_llm.bind_tools([answer_the_question]).invoke(messages)
        ai_msg = proposal_llm.invoke(messages)

        answer_info = retrieve_single_answer_info(ai_msg)
        new_reactions.append(strip_out_thinking_process(answer_info.response))

        # if not ai_msg.tool_calls:
        #     print("No Tool Calls", ai_msg.content)
        #     new_reactions.append("")  # TODO: maybe change this
        # else:
        #     new_reactions.append(ai_msg.tool_calls[0]["args"]["answer"])

    return Reaction(question=question_about_proposal, reactions=new_reactions)


def get_final_decision(
    committee_llm: BaseChatModel,
    original_question: str,
    proposals: list[Proposal],
    reactions: list[Reaction],
):
    messages = (
        [
            SystemMessage(
                "You are part of a committee of a university to answer pressuring questions of significant importance."
                "Listen carefully to the conversation and give your decision on what is the correct answer after"
                "evaluating the different viewpoints, discussions, facts and assumptions formulated."
            ),
            HumanMessage(original_question),
        ]
        + [
            HumanMessage(
                f"Proposal {i}: The correct answer is ({p.answer}) {get_answer(original_question, p.answer)}. {p.reasoning}"
            )
            for i, p in enumerate(proposals)
        ]
        + [
            x
            for r in reactions
            for x in [
                HumanMessage(r.question),
                *[
                    HumanMessage(f"Proposal {i}: {a}")
                    for i, a in enumerate(r.reactions)
                ],
            ]
        ]
        + [
            HumanMessage(
                "After the discussion, what do you think is the correct solution?"
                "Please evaluate carefully and submit your decision using the given tool."
            )
        ]
    )

    @tool(return_direct=True)
    def submit_solution_opinion(correct_answer: str):
        """
        Submit the answer to the question that you think is correct.
        Evaluate the given discussion and find what you think is the most senseful answer.
        You do not have to follow any opinions, but make up your own.
        Do give the answer as: (X).
        """
        pass

    ai_msg = committee_llm.bind_tools([submit_solution_opinion]).invoke(messages)

    try:
        return parse_tool_call_arguments(ai_msg)["correct_answer"]
    except KeyError as e:
        raise InvalidToolCallException() from e


@register_decision_scheme("tribal-council")
class TribalCouncilDebate(DecisionScheme[TribalCouncilConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        with phoenix_log_span("collecting proposals", title="Proposal Round"):
            proposals = self.proposal_round(
                example_input.question, self.config.num_different_proposals
            )
            if len(proposals) < self.config.num_different_proposals:
                with phoenix_log_span(
                    "Could not retrieve enough different proposals due to invalid formulations, etc.",
                    title="Aborting",
                ):
                    return ExampleOutput(
                        # TODO:
                        used_input_tokens=0,
                        used_output_tokens=0,
                        final_answer=None,
                        answers_at_end=[],
                        answers_at_beginning=[str(p.answer) for p in proposals],
                        history=[],  # TODO:
                    )

        with phoenix_log_span("asking questions", title="Question Round"):
            reactions = self.question_round(example_input.question, proposals)

        with phoenix_log_span("retrieving member opinions", title="Opinion Round"):
            final_answers = self.opinion_round(
                example_input.question, proposals, reactions
            )

        return ExampleOutput(  # TODO:
            used_input_tokens=0,
            used_output_tokens=0,
            final_answer=None,
            answers_at_end=final_answers,
            answers_at_beginning=[str(p.answer) for p in proposals],
            history=[],  # TODO:
        )

    def proposal_round(self, question: str, num_different_proposals: int):
        proposals: list[Proposal] = []
        proposal_llm = get_llm(
            self.config.proposal_agent.llm, self.config.proposal_agent.backend
        )

        n_same_proposals = 0

        while len(proposals) < num_different_proposals:
            try:
                if (new_prop := make_proposal(proposal_llm, question)) is None:
                    continue
            except (NoToolCallsException, InvalidToolCallException):
                n_same_proposals += 1
                if n_same_proposals > self.config.accept_after_n_invalid_proposals:
                    return proposals
                continue

            if (
                not any(p.answer == new_prop.answer for p in proposals)
                or n_same_proposals >= self.config.accept_after_n_invalid_proposals
            ):
                n_same_proposals = 0
                proposals.append(new_prop)
            else:
                n_same_proposals += 1

        return proposals

    def question_round(
        self, question: str, proposals: list[Proposal]
    ) -> list[Reaction]:
        council_members = [
            get_llm(self.config.proposal_agent.llm, self.config.proposal_agent.backend)
            for _ in range(self.config.council_size)
        ]

        proposal_llm = get_llm(
            self.config.proposal_agent.llm, self.config.proposal_agent.backend
        )

        reactions = []
        invalid_questions_received = 0
        i = 0

        while i < len(council_members):
            with phoenix_log_span(f"Question {i}", title=f"Question {i}"):
                try:
                    question_about_proposal = form_question_about_proposal(
                        council_members[i], question, proposals, reactions
                    )
                    i += 1

                except (NoToolCallsException, InvalidToolCallException):
                    invalid_questions_received += 1
                    if (
                        invalid_questions_received
                        > self.config.total_invalid_question_formulations_accepted
                    ):
                        with phoenix_log_span(
                            f"Too many invalid formulations received in one round (> {self.config.total_invalid_question_formulations_accepted})",
                            title="Abort Question Round",
                        ):
                            return reactions
                    continue

            with phoenix_log_span(f"Answers {i}", title=f"Answer {i}"):
                reaction = answer_question_about_proposal(
                    proposal_llm,
                    question,
                    proposals,
                    reactions,
                    question_about_proposal,
                )
            reactions.append(reaction)

        return reactions

    def opinion_round(
        self, question: str, proposals: list[Proposal], reactions: list[Reaction]
    ) -> list[str]:
        council_members = [
            get_llm(self.config.proposal_agent.llm, self.config.proposal_agent.backend)
            for _ in range(self.config.council_size)
        ]
        decisions = []

        i = 0
        total_invalid_decisions = 0

        while i < len(council_members):
            try:
                decision = get_final_decision(
                    council_members[i], question, proposals, reactions
                )
                decisions.append(decision)
                i += 1
            except (NoToolCallsException, InvalidToolCallException):
                total_invalid_decisions += 1
                if (
                    total_invalid_decisions
                    > self.config.total_invalid_final_decisions_accepted
                ):
                    with phoenix_log_span(
                        f"Too many invalid formulations received in one round (> {self.config.total_invalid_final_decisions_accepted})",
                        title="Abort Question Round",
                    ):
                        return decisions

        return decisions

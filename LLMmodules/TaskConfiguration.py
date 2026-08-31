# Description: This file contains the configuration for the tasks.
# Task List -----------------------------------------------------
# 1. Catchup task
catchup = {
    'path': "LLMmodules/Prompts/CACC/catchup.txt",
    'reward': "LLMmodules/Prompts/CACC/catchup_reward.txt",
}

slowdown = {
    'path': "LLMmodules/Prompts/CACC/slowdown.txt",
}

pandemic = {
    'path': "LLMmodules/Prompts/Pandemic/pandemic.txt",
}

task_step = {
    'proposal': 'LLMmodules/Prompts/Instruction/4_Proposal_multi.txt',
    'evaluate': 'LLMmodules/Prompts/Instruction/5_EvaluateProposal.txt',
    'rules': 'LLMmodules/Prompts/Instruction/7_SummarizeRule.txt',
    'get_errors': 'LLMmodules/Prompts/Instruction/10_GetErrors.txt',
    'update_rules':  'LLMmodules/Prompts/Instruction/9_UpdateRules.txt',
    'repropose': 'LLMmodules/Prompts/Instruction/11_Reproposal.txt',
    'gradient': 'LLMmodules/Prompts/Instruction/12_GenGradient.txt',
    'self_reflection': 'LLMmodules/Prompts/Instruction/13_SelfReflection.txt',
}

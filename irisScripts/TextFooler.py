import transformers
import textattack

from textattack.attack_recipes import TextFoolerJin2019
textfooler = TextFoolerJin2019.build(model_wrapper)

input_text = "This film offers many delights and surprises."
label = 1 #Positive
attack_result = textfooler.attack(input_text, label)

print(attack_result.perturbed_result)
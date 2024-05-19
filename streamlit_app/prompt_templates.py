en_prompt_template =""" 
<s>[INST] You are an expert agent in information extraction and summarization.
Given a meeting transcription generate a summary for record-keeping. 

FORMAT:
Title: 
Key points:  
Key Decisions:  
Action Items: 

The meeting transcript is as follows:
----------------
{text}
----------------
Focus solely on the information presented in the transcript. 
Aim for clarity and brevity to ensure the summary is useful for attendees and those unable to attend.
VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text. [/INST]
"""
en_refine_template = """
<s>[INST] Your task is to refine the initial summary based on additional context.
Below is the existing summary:
{existing_answer}
Now, we'll provide more context from the meeting transcript:
----------------
{text}
----------------
Refine the summary to incorporate the new context, or maintain the original if the context does not add value. 
FORMAT:
Title: 
Key points:
Key Decisions: 
Action Items: 

Aim for clarity and brevity.
VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text. [/INST]
"""


no_prompt_template = """
Gi meg ett sammendrag av dette møte, basert på transkripsjonen. 
Møtetranskripsjonen er som følger:       
--------        
{text}
--------
Skriv på norsk og bruk følgende format:
Tittel
- Hovedpunkter:
- Viktige beslutninger:
- Handlingspunkter:

Avslutt med merknad om at sammendraget er automatisk generert.
SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv møtesammendraget. Ingen intro, ingen annen tekst.
"""
no_refine_template = """
Din oppgave er å forbedre det innledende sammendraget basert på ytterligere kontekst.
Nedenfor er det eksisterende sammendraget:
{existing_answer}
Nå vil vi gi mer kontekst fra møtetranskripsjonen:
----------------
{text}
----------------
Skriv på norsk og bruk følgende format:
Tittel
- Hovedpunkter:
- Viktige beslutninger:
- Handlingspunkter:

Opdater sammendraget slik at det bruker den nye informasjonen, eller behold det originale hvis den nye informasjonen ikke tilfører verdi.
"""

king_prompt_template = """
<s>[INST] Du er en ekspertagent i informasjonsutvinning og sammendrag.
Gitt en tale transkripsjon, generer et sammendrag for arkivering.

Transkripsjonen er som følger:
----------------
{text}
----------------
Fokuser kun på informasjonen presentert i transkripsjonen.
Sikte på klarhet og korthet for å sikre at sammendraget er nyttig. Skriv på norsk.
SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv sammendraget. Ingen intro, ingen annen tekst. [/INST]
"""
king_refine_template = """
<s>[INST] Din oppgave er å forbedre dette sammendraget basert på ytterligere kontekst.
Nedenfor er det eksisterende sammendraget:
{existing_answer}
Her er mer informasjon som kan hjelpe deg med å forbedre sammendraget:
----------------
{text}
----------------
Opdater sammendraget slik at det bruker den nye informasjonen, eller behold det originale hvis den nye informasjonen ikke tilfører verdi.
Sikte på klarhet og korthet. Skriv på norsk.
SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv sammendraget. Ingen intro, ingen annen tekst [/INST]
"""

chatml_en_prompt_template =""" 
<|im_start|>user 
You are an expert agent in information extraction and summarization.
Given a meeting transcription generate a summary for record-keeping. 

FORMAT:
Title: 
Key points:  
Key Decisions:  
Action Items: 

The meeting transcript is as follows:
----------------
{text}
----------------
Focus solely on the information presented in the transcript. 
Aim for clarity and brevity to ensure the summary is useful for attendees and those unable to attend.
VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text. <|im_end|>
<|im_start|>assistant
"""
chatml_en_refine_template = """
<|im_start|>user 
Your task is to refine the initial summary based on additional context.
Below is the existing summary:
{existing_answer}
Now, we'll provide more context from the meeting transcript:
----------------
{text}
----------------
Refine the summary to incorporate the new context, or maintain the original if the context does not add value. 
FORMAT:
Title: 
Key points:
Key Decisions: 
Action Items: 

Aim for clarity and brevity.
VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text. <|im_end|>
<|im_start|>assistant
"""

chatml_no_prompt_template = """
<|im_start|>user
Du er en ekspertagent i informasjonsutvinning og sammendrag.
Gitt en møtetranskripsjon, generer et sammendrag for arkivering.

FORMAT:
Tittel:
Hovedpunkter:
Viktige beslutninger:
Handlingspunkter:

Møtetranskripsjonen er som følger:
----------------
{text}
----------------
Fokuser kun på informasjonen presentert i transkripsjonen. Skriv på norsk.
Sikte på klarhet og korthet for å sikre at sammendraget er nyttig for deltakere og de som ikke kunne delta.
SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv møtesammendraget. Ingen intro, ingen annen tekst. <|im_end|>
<|im_start|>assistant
"""
chatml_no_refine_template = """
<|im_start|>user
Din oppgave er å forbedre det innledende sammendraget basert på ytterligere kontekst.
Nedenfor er det eksisterende sammendraget:
{existing_answer}
Nå vil vi gi mer kontekst fra møtetranskripsjonen:
----------------
{text}
----------------
FORMAT:
Tittel:
Hovedpunkter:
Viktige beslutninger:
Handlingspunkter:

Opdater sammendraget slik at det bruker den nye informasjonen, eller behold det originale hvis den nye informasjonen ikke tilfører verdi.
Sikte på klarhet og korthet. Skriv på norsk.
SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv sammendraget. Ingen intro, ingen annen tekst. <|im_end|>
<|im_start|>assistant
"""
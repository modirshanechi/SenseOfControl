function create_consent_form(){
    consent_text = `<p style='font-size:30px'>Participation in the Learning and Cognitive Control Study</p>`
    consent_text += `<p>This is a psychology experiment conducted by Dr. Peter Dayan, director of the Max Planck Institute for Biological Cybernetics, and the members of his lab.</p>`
    consent_text += `<p></p>`
    consent_text += `<p>All data collected will be anonymous. We will not ask for any additional personally identifying information and will handle responses as confidentially as possible. However, we cannot guarantee the confidentiality of information transmitted over the Internet. We will keep de-identified data collected as part of this experiment indefinitely, and such data may be used as part of future studies and/or made available to the wider research community for follow-up analyses. Data used in scientific publications will remain completely anonymous.</p>`
    consent_text += `<p></p>`
    consent_text += `<p>Your participation in this research is voluntary. You may refrain from answering any questions that make you uncomfortable and may withdraw your participation at any time without penalty by exiting this task. You may choose not to complete certain parts of the task or answer certain questions.</p>`
    consent_text += `<p></p>`
    consent_text += `<p>Other than monetary compensation, participating in this study will provide no direct benefits to you. However, we hope that this research will benefit society at large by contributing towards establishing a scientific foundation for improving people's learning and cognitive control abilities.</p>`
    consent_text += `<p></p>`
    consent_text += `<p>By pressing the button below you consent to taking part in this study.</p>`
    return consent_text
}
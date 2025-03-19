---
name: "\U0001F392 Onboarding new users" 
about: Onboarding new users
title: ''
labels: 'onboarding'
assignees: ''

---
	
## :school_satchel: Onboarding
<!--Add a welcome message tagging github username -->
Welcome @ username to `unetvit4sclera` :tada:	 

We recommend to cover the following points for your onboarding. 
Some of these might not be relevant, so feel free to skip them and add questions in this issue.
* [ ] Clone repo [:link:](https://github.com/oocular/unetvit4sclera/tree/main?tab=readme-ov-file#octocat-cloning-repository).
* [ ] Install uv and create virtual environment [:link:](https://github.com/oocular/unetvit4sclera/tree/main/docs).
* [ ] Run bash scripts for:
    * `bash scripts/activate_pre_commit.bash`
    * `bash scripts/tests/unit_test_unet_pipeline.bash`
* [ ] Familiarise yourself with the repository and organise a code review session with a colleague.
* [ ] Request SBVPI dataset to Matej Vitek by filling a form as shonwn [here](https://github.com/oocular/unetvit4sclera/tree/main/data/licence-agreements)
* [ ] Modify `bash scripts/models/train_unet_with_mobious.bash` and related scripts to use sbvpi datasets.
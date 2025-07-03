Dataset Details:

Dataset contains recordings from 36 speakers (18 Male & 18 Female) belonging to students of Grade 3 who have each been tested on 1 English & 1 Hindi story. There are 3 different stories for each language (Total 6 stories) of which 1 is randomly chosen for testing of a particular student. Each story has 2 paragraphs that were read by the student. Details of the story that was read by the child and the gender of the child is given in the RecordingDetails.csv file. Since there are 36 speakers who have read 2 stories (1 English & 1 Hindi) containing 2 paras each, we have 144 recordings in the dataset.

Filenaming convention:
StudentID_TimeStampInEpoch_ParaNoTakeNo.m4a
- StudentID is a unique identifier for every student
- TimeStampInEpoch contains the timestamp in epoch of the attempt of the student. This is the same for both paras for a particular language.
- Para No can be 2 or 3 based on whether the audio contains the second or third para of the story. Para 1 contains the title of the story and is not used.
- Take No is the Take for a particular paragraph. Sometimes there could be a problem while recording and the teacher can re-record a particular para. Each re-recording attempt is considered as a fresh take.
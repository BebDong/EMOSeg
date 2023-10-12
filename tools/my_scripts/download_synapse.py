import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login('BebDong', '19970628')
files = synapseutils.syncFromSynapse(syn, 'syn3379050', path='/cluster/work/cvl/qutang/')
